
# frozen_string_literal: true

# ============================================================================
# Peer Architecture for Lightweight Symbolic Execution
# S-expression (Scheme) 版本 — Ruby 实现
#
# 论文: Bruni, Disney, Flanagan — UC Santa Cruz
#
# 架构:
#   Scheme 源码 → SXP 解析 → 符号求值器 → Z3 约束求解
#        ↑ Figure 5          ↑ Figure 6      ↑ Figure 4
#    IntProxy / BoolProxy   test() 回溯循环   SMT 接口
#
# 依赖: gem install sxp z3
# ============================================================================

require 'sxp'
require 'z3'

# ============================================================================
# 第 1 层：Scheme 预处理 & SXP 解析
# ============================================================================

def preprocess_scheme(src)
  src = src.gsub('#t', 'true').gsub('#f', 'false')
  src.lines.map { |l| l.sub(/;.*/, '') }.join("\n")
end

def parse_scheme(src)
  src = preprocess_scheme(src)
  # SXP 一次只解析一个顶层 S 表达式，多个表达式包装为 (begin ...)
  ast = SXP.read("(begin #{src})")
  # ast = [:begin, expr1, expr2, ...]
  ast.is_a?(Array) && ast.first == :begin ? ast[1..] : [ast]
end

# ============================================================================
# 第 2 层：符号值 — SymInt / SymBool (论文 Figure 5 IntProxy / BoolProxy)
# ============================================================================

# 符号整数：包装 Z3::IntExpr 或 Ruby Integer（常量）
# Ruby Z3 自动将 Integer 转型为 Z3 表达式，无需手动 IntVal
class SymInt
  attr_reader :z3_expr

  def initialize(expr)
    @z3_expr = expr  # Z3::IntExpr 或 Integer
  end

  def +(other)  = SymInt.new(_bin(other) { |a, b| a + b })
  def -(other)  = SymInt.new(_bin(other) { |a, b| a - b })
  def *(other)  = SymInt.new(_bin(other) { |a, b| a * b })
  def /(other)  = SymInt.new(_bin(other) { |a, b| a / b })
  def %(other)  = SymInt.new(_bin(other) { |a, b| a % b })
  def -@        = SymInt.new(-@z3_expr)

  def ==(other) = SymBool.new(_bin(other) { |a, b| a == b })
  def !=(other) = SymBool.new(_bin(other) { |a, b| a != b })
  def <(other)  = SymBool.new(_bin(other) { |a, b| a < b })
  def >(other)  = SymBool.new(_bin(other) { |a, b| a > b })
  def <=(other) = SymBool.new(_bin(other) { |a, b| a <= b })
  def >=(other) = SymBool.new(_bin(other) { |a, b| a >= b })

  def inspect = @z3_expr.is_a?(Integer) ? "SymInt(#{@z3_expr})" : "SymInt(#{@z3_expr})"
  alias to_s inspect

  private

  def _bin(other)
    a = @z3_expr
    b = other.is_a?(SymInt) ? other.z3_expr : other
    yield a, b
  end
end

# 符号布尔值：包装 Z3::BoolExpr 或 Ruby true/false
class SymBool
  attr_reader :z3_expr

  def initialize(expr)
    @z3_expr = expr  # Z3::BoolExpr 或 true/false
  end

  def inspect = "SymBool(#{@z3_expr})"
  alias to_s inspect
end

# ============================================================================
# 第 3 层：词法作用域环境
# ============================================================================

class Env
  def initialize(parent = nil)
    @parent = parent
    @bindings = {}
  end

  def lookup(name)
    key = name.to_s
    if @bindings.key?(key)
      @bindings[key]
    elsif @parent
      @parent.lookup(key)
    else
      v = SymInt.new(Z3::Int(key))
      @bindings[key] = v
      v
    end
  end

  def define(name, value)
    @bindings[name.to_s] = value
  end

  def copy
    new_env = Env.new(@parent&.copy)
    new_env.instance_variable_set(:@bindings, @bindings.dup)
    new_env
  end
end

# ============================================================================
# 第 4 层：全局路径状态 (论文 Figure 6 lines 1-2)
# ============================================================================

$path = []
$path_condition = []
$MAX_DEPTH = 10

class DepthExceeded < StandardError; end
class AssertionFailure < StandardError; end

# ============================================================================
# 第 5 层：S 表达式符号求值器 (论文 Figure 6 核心算法)
# ============================================================================

class SExprEvaluator
  def initialize(env = Env.new)
    @env = env
  end

  # ---- 主入口 ----
  def eval(expr)
    case expr
    when Integer      then SymInt.new(expr)
    when Float        then SymInt.new(expr.to_i)
    when true, false  then expr
    when Symbol
      case expr.to_s
      when 'true'  then true
      when 'false' then false
      else @env.lookup(expr.to_s)
      end
    when Array
      return SymInt.new(0) if expr.empty?
      op = expr.first
      op_str = op.is_a?(Symbol) ? op.to_s : op.to_s
      dispatch(op_str, expr[1..])
    else
      raise "无法求值: #{expr.inspect}"
    end
  end

  private

  def dispatch(op, args)
    case op
    when 'if'       then eval_if(args)
    when 'define'   then eval_define(args)
    when 'lambda'   then eval_lambda(args)
    when 'begin'    then eval_begin(args)
    when 'let'      then eval_let(args)
    when 'assert'   then eval_assert(args)
    when '+', '-', '*', '/', '%'           then eval_arithmetic(op, args)
    when '=', '<', '>', '<=', '>=', '/='   then eval_comparison(op, args)
    when 'and', 'or', 'not'                then eval_boolean(op, args)
    else eval_call(op, args)
    end
  end

  # ========== if — 分支点 (论文 5.4 节, Figure 6 lines 29-53) ==========

  def eval_if(args)
    raise SyntaxError, "if 需要 3 个参数" unless args.length == 3
    cond_expr, then_expr, else_expr = args

    cond_val = eval(cond_expr)

    # 抽取 Z3 布尔公式
    formula = extract_formula(cond_val)

    # Step 1-2: 可行性检查 (论文 lines 32-38)
    true_feasible  = solve_with(formula)
    false_feasible = solve_with(negate_formula(formula))

    # 强制分支: 仅一侧可行 (不推送永真/永假约束)
    if true_feasible && !false_feasible
      $path_condition.push(formula) unless formula.equal?(true) || formula.equal?(false)
      return eval(then_expr)
    end
    if false_feasible && !true_feasible
      neg = negate_formula(formula)
      $path_condition.push(neg) unless neg.equal?(true) || neg.equal?(false)
      return eval(else_expr)
    end

    # 自由分支: 两侧都可行

    # Step 3: 按预定路径走 (论文 lines 40-45)
    if $path.length > $path_condition.length
      branch = $path[$path_condition.length]
      constraint = branch ? formula : negate_formula(formula)
      $path_condition.push(constraint)
      return branch ? eval(then_expr) : eval(else_expr)
    end

    # Step 4: 深度限制 (论文 line 48)
    if $path.length >= $MAX_DEPTH
      raise DepthExceeded, "深度超过 #{$MAX_DEPTH}"
    end

    # Step 5: 默认走 true (论文 lines 51-53)
    $path.push(true)
    $path_condition.push(formula)
    eval(then_expr)
  end

  # ========== define / lambda / begin / let ==========

  def eval_define(args)
    raise SyntaxError, "define 需要 2 个参数" unless args.length == 2
    var_expr, val_expr = args

    # 函数定义: (define (f x y) body)
    if var_expr.is_a?(Array)
      func_name = var_expr.first.to_s
      func_args = var_expr[1..].map(&:to_s)
      closure_env = @env.copy
      closure = { type: :closure, args: func_args, body: val_expr, env: closure_env }
      @env.define(func_name, closure)
      closure_env.define(func_name, closure)  # 支持递归
      return closure
    end

    var_name = var_expr.to_s
    val = eval(val_expr)
    @env.define(var_name, val)
    val
  end

  def eval_lambda(args)
    raise SyntaxError, "lambda 需要 (params) body" unless args.length == 2
    param_names = args[0].map(&:to_s)
    { type: :closure, args: param_names, body: args[1], env: @env.copy }
  end

  def eval_begin(args)
    result = SymInt.new(0)
    args.each { |e| result = eval(e) }
    result
  end

  def eval_let(args)
    raise SyntaxError, "let 需要 (bindings) body" unless args.length >= 2
    bindings, *body_exprs = args

    let_env = Env.new(@env)
    bindings.each do |b|
      let_env.define(b.first.to_s, eval(b[1]))
    end

    saved_env = @env
    @env = let_env
    result = body_exprs.length == 1 ? eval(body_exprs.first) : eval_begin(body_exprs)
    @env = saved_env
    result
  end

  # ========== assert — 断言 (论文 Section 6 @ensure) ==========

  def eval_assert(args)
    cond_val = eval(args.first)
    formula = extract_formula(cond_val)

    # 检查: PC ∧ ¬condition 是否可满足?
    if solve_with(negate_formula(formula))
      model = get_counterexample(negate_formula(formula))
      raise AssertionFailure,
        "断言失败: #{serialize(args.first)}\n" \
        "  路径条件: #{pc_display}\n" \
        "  反例: #{model}"
    end
    true
  end

  # ========== 算术 / 比较 / 布尔 ==========

  def eval_arithmetic(op, args)
    vals = args.map { |a| eval(a) }
    result = vals.reduce do |acc, v|
      case op
      when '+' then acc + v
      when '-' then acc - v
      when '*' then acc * v
      when '/' then acc / v
      when '%' then acc % v
      end
    end
    result || vals.first
  end

  def eval_comparison(op, args)
    raise SyntaxError, "#{op} 需要 2 个参数" unless args.length == 2
    left  = eval(args[0])
    right = eval(args[1])

    case op
    when '='  then left == right
    when '<'  then left < right
    when '>'  then left > right
    when '<=' then left <= right
    when '>=' then left >= right
    when '/=' then left != right
    end
  end

  def eval_boolean(op, args)
    case op
    when 'and'
      result = true
      args.each { |a| result = bool_and(result, eval(a)) }
      result
    when 'or'
      result = false
      args.each { |a| result = bool_or(result, eval(a)) }
      result
    when 'not'
      raise SyntaxError, "not 需要 1 个参数" unless args.length == 1
      bool_not(eval(args.first))
    end
  end

  # ========== 函数调用 ==========

  def eval_call(op_str, args)
    func = @env.lookup(op_str)
    arg_vals = args.map { |a| eval(a) }

    unless func.is_a?(Hash) && func[:type] == :closure
      raise TypeError, "#{op_str} 不是可调用函数: #{func.inspect}"
    end

    raise ArgumentError,
      "#{op_str}: 期望 #{func[:args].length} 个参数, 实际 #{arg_vals.length}" \
      unless arg_vals.length == func[:args].length

    call_env = Env.new(func[:env])
    func[:args].each_with_index { |name, i| call_env.define(name, arg_vals[i]) }

    saved_env = @env
    @env = call_env
    result = eval(func[:body])
    @env = saved_env
    result
  end

  # ========== Z3 辅助方法 ==========

  def current_pc
    return nil if $path_condition.empty?
    # 过滤掉 Ruby true (无约束) 和 false (永假，不应出现)
    real_constraints = $path_condition.reject { |c| c.equal?(true) || c.equal?(false) }
    return nil if real_constraints.empty?
    real_constraints.reduce { |a, b| a & b }
  end

  # 抽取 Z3 布尔公式 (可能返回 Z3::BoolExpr 或 Ruby true/false)
  # 注意: Z3 重载了 ==, 不能用 val == true/false 比较 (会返回 Z3 表达式)
  def extract_formula(val)
    if val.is_a?(SymBool)
      val.z3_expr
    elsif val.equal?(true)
      true
    elsif val.equal?(false)
      false
    elsif val.is_a?(SymInt)
      val.z3_expr != 0
    else
      raise TypeError, "无法转换为布尔公式: #{val.inspect}"
    end
  end

  # 公式取反
  def negate_formula(f)
    return false if f.equal?(true)
    return true  if f.equal?(false)
    !f
  end

  # 检查 PC ∧ formula 是否可满足
  def solve_with(formula)
    return true  if formula.equal?(true)
    return false if formula.equal?(false)
    s = Z3::Solver.new
    pc = current_pc
    s.assert(pc) if pc
    s.assert(formula)
    s.satisfiable?
  end

  def get_counterexample(violation_formula)
    return {} if violation_formula.equal?(false)
    s = Z3::Solver.new
    pc = current_pc
    s.assert(pc) if pc
    s.assert(violation_formula) unless violation_formula.equal?(true)
    return {} unless s.satisfiable?

    model = s.model
    h = {}
    model.to_s.scan(/(\w+)\s*->\s*(-?\d+)/).each do |name, val|
      h[name] = val.to_i
    end
    h
  end

  def pc_display
    pc = current_pc
    pc ? pc.to_s : '#t'
  end

  # ---- 布尔运算辅助 (注意: Z3 重载 ==, 必须用 equal?) ----
  def bool_and(a, b)
    return false                 if a.equal?(false) || b.equal?(false)
    return b                     if a.equal?(true)
    return a                     if b.equal?(true)
    SymBool.new(a.z3_expr & b.z3_expr)
  end

  def bool_or(a, b)
    return true                  if a.equal?(true) || b.equal?(true)
    return b                     if a.equal?(false)
    return a                     if b.equal?(false)
    SymBool.new(a.z3_expr | b.z3_expr)
  end

  def bool_not(v)
    return true                  if v.equal?(false)
    return false                 if v.equal?(true)
    SymBool.new(!v.z3_expr)
  end
end

# ============================================================================
# 序列化辅助
# ============================================================================

def serialize(expr)
  case expr
  when Symbol then expr.to_s
  when Array  then "(#{expr.map { |e| serialize(e) }.join(' ')})"
  when Integer then expr.to_s
  when true, false then expr.to_s
  else expr.to_s
  end
end

# ============================================================================
# 第 6 层：模型检查主循环 (论文 Figure 6 lines 6-22)
# ============================================================================

def symbolic_test(program_text, label: nil)
  puts label if label
  ast = parse_scheme(program_text)

  # 分离断言和程序体
  assertions = []
  body_exprs  = []
  ast.each do |e|
    if e.is_a?(Array) && e.first.to_s == 'assert'
      assertions << e
    else
      body_exprs << e
    end
  end

  errors = []

  # 路径探索
  unless body_exprs.empty?
    body = body_exprs.length == 1 ? body_exprs.first : [:begin, *body_exprs]
    errors += explore_paths(body)
  end

  # 断言检查 (在纯符号环境下验证)
  unless assertions.empty?
    errors += check_assertions(assertions, body_exprs)
  end

  errors
end

def explore_paths(body)
  puts '-' * 44
  puts '路径探索 (论文 Figure 6 回溯循环):'
  puts '-' * 44

  errors = []
  iteration = 0

  loop do
    $path_condition = []
    iteration += 1

    begin
      result = SExprEvaluator.new.eval(body)
      puts "[iter #{iteration}] path=#{trunc($path.inspect, 50)}  " \
           "result=#{trunc(result.inspect, 40)}"
    rescue DepthExceeded
      puts "[iter #{iteration}] path=#{trunc($path.inspect, 50)}  PRUNED (depth limit)"
    rescue AssertionFailure => e
      puts "[iter #{iteration}] path=#{trunc($path.inspect, 50)}  ✗ ASSERT FAIL"
      errors << e.message
    rescue StandardError => e
      puts "[iter #{iteration}] path=#{trunc($path.inspect, 50)}  ERROR: #{e.message}"
    end

    # 回溯 (论文 lines 16-22)
    $path.pop while $path.length > 0 && !$path.last
    break if $path.empty?
    $path[-1] = false
  end

  puts "共 #{iteration} 条路径。"
  errors
end

def check_assertions(assertions, body_exprs)
  puts '-' * 44
  puts '断言检查 (论文 Section 6 契约系统):'
  puts '-' * 44

  errors = []
  global_env = Env.new

  # 先执行所有定义
  body_exprs.each do |expr|
    SExprEvaluator.new(global_env).eval(expr)
  end

  assertions.each do |assert_expr|
    # 每条断言使用独立的路径状态
    $path = []
    $path_condition = []

    begin
      evaluator = SExprEvaluator.new(global_env)
      evaluator.send(:eval_assert, assert_expr[1..])
      puts "  ✓ #{serialize(assert_expr)}"
    rescue AssertionFailure => e
      puts "  ✗ #{e.message}"
      errors << e.message
    rescue StandardError => e
      puts "  ? #{serialize(assert_expr)} — #{e.message}"
    end
  end
  errors
end

def trunc(s, max_len)
  s.length <= max_len ? s : s[0...max_len] + '...'
end

# ============================================================================
# 演示
# ============================================================================

if __FILE__ == $PROGRAM_NAME
  puts '=' * 60
  puts 'Peer Architecture for Lightweight Symbolic Execution'
  puts 'Bruni, Disney, Flanagan — S-expression (Ruby) 复现'
  puts '=' * 60

  # ---- 示例 1: abs (论文 Figure 2) ----
  puts '', '█ 示例 1: abs 绝对值 — 论文 Figure 2'
  $path = []
  errs = symbolic_test(<<~'SCHEME', label: nil)
    (define (abs x)
      (if (>= x 0)
          x
          (- 0 x)))

    (assert (= (abs 5) 5))
    (assert (= (abs -3) 3))
    (assert (>= (abs x) 0))
  SCHEME
  puts errs.empty? ? '  ✓ 全部通过' : '  ✗ 存在失败'

  # ---- 示例 2: succ 含 bug (论文 Figure 2) ----
  puts '', '█ 示例 2: succ 含 bug (x=42768) — 论文 Figure 2'
  $path = []
  errs = symbolic_test(<<~'SCHEME', label: nil)
    (define (succ x)
      (if (= x 42768)
          99999
          (+ x 1)))

    (assert (not (= (succ 42768) 42769)))
  SCHEME
  puts errs.empty? ? '  (断言通过 — 函数确实返回非期望值)' : '  ✓ 断言失败，检测到 bug'

  # ---- 示例 3: factorial 含 bug (论文 Figure 9) ----
  puts '', '█ 示例 3: factorial — 论文 Figure 9'
  $path = []
  errs = symbolic_test(<<~'SCHEME', label: nil)
    (define (fact x)
      (if (= x 40)
          123456789
          (if (<= x 1)
              1
              (* x (fact (- x 1))))))

    (assert (= (fact 5) 120))
    (assert (= (fact 40) 815915283247897734345611269596115894272000000000))
  SCHEME
  puts errs.empty? ? '  ✓ 全部通过' : '  ✗ 存在失败 (符合预期: x=40 有 bug)'

  # ---- 示例 4: 自由变量 4 条路径 ----
  puts '', '█ 示例 4: 嵌套 if (自由变量 x, y, 4 条路径)'
  $path = []
  errs = symbolic_test(<<~'SCHEME', label: nil)
    (if (> x 5)
        (if (= y 10)
            1
            2)
        (if (< y 0)
            3
            4))
  SCHEME
  puts errs.empty? ? '  ✓ 全部通过' : '  ✗ 存在失败'

  # ---- 示例 5: max + 符号断言 ----
  puts '', '█ 示例 5: max 函数 + 符号断言 (∀a,b: max(a,b)≥a)'
  $path = []
  errs = symbolic_test(<<~'SCHEME', label: nil)
    (define (max2 a b)
      (if (> a b) a b))

    (assert (>= (max2 10 5) 10))
    (assert (>= (max2 3 7) 7))
    (assert (>= (max2 a b) a))
    (assert (>= (max2 a b) b))
  SCHEME
  puts errs.empty? ? '  ✓ 全部通过' : '  ✗ 存在失败'

  puts '', '=' * 60
end
