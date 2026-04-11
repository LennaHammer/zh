require 'sxp'
require 'z3'

# -----------------------------------------------------------------------------
# 1. 替换函数 Q[e/x]
# -----------------------------------------------------------------------------
def substitute(expr, x, e)
  case expr
  when Symbol then
    expr == x ? e : expr
  when Array then
    expr.map { |child| substitute(child, x, e) }
  else
    expr
  end
end

# -----------------------------------------------------------------------------
# 2. S表达式 转 Z3 表达式
# -----------------------------------------------------------------------------
def sxp_to_z3(expr)
  case expr
  when Integer
    expr
  when Symbol
    Z3.Int(expr.to_s)
  when Array
    op, *args = expr
    z3_args = args.map { |a| sxp_to_z3(a) }
    case op
    when :+ then z3_args.reduce(:+)
    when :- then z3_args.reduce(:-)
    when :* then z3_args.reduce(:*)
    when :'/' then z3_args.reduce(:/)
    when :> then z3_args[0] > z3_args[1]
    when :>= then z3_args[0] >= z3_args[1]
    when :< then z3_args[0] < z3_args[1]
    when :<= then z3_args[0] <= z3_args[1]
    when :== then z3_args[0] == z3_args[1]
    when :'=' then z3_args[0] == z3_args[1]
    when :!= then z3_args[0] != z3_args[1]
    when :implies then Z3.Implies(z3_args[0], z3_args[1])
    when :and then z3_args.reduce(:&)
    when :'||' then z3_args.reduce(:|)
    when :not then !z3_args[0]
    else raise "不支持运算符: #{op}"
    end
  else
    expr
  end
end
 

# -----------------------------------------------------------------------------
# 3. 最弱前置条件 WP 计算（核心）
# -----------------------------------------------------------------------------

def wp(stmt, post, vcs)
  case stmt
  in [:skip]
    post
  in [:set, x, e]
    # wp(x := e, Q) = Q[e/x]
    substitute(post, x, e)
  in [:seq, s1, s2]
    # wp(S1;S2, Q) = wp(S1, wp(S2, Q))
    wp(s1, wp(s2, post, vcs), vcs)
  in [:seq, *stmts]
    # wp(S1;S2;...;Sn, Q) = wp(S1, wp(S2, ..., wp(Sn, Q)...))
    stmts.reverse.reduce(post) { |acc, s| wp(s, acc, vcs) }
  in [:if, b, s1, s2]
    # wp(if b then S1 else S2, Q) = (b → wp(S1,Q)) ∧ (¬b → wp(S2,Q))
    wp1 = wp(s1, post, vcs)
    wp2 = wp(s2, post, vcs)
    [:and, [:implies, b, wp1], [:implies, [:not, b], wp2]]
  in [:while, b, inv, body]
    # wp(while b do S, Q) = I
    # 其中 I 是循环不变式，需要用户提供
    # 生成验证条件：
    # 1. 不变式初始化：inv
    # 2. 归纳：inv ∧ b → wp(body, inv)
    # 3. 退出：inv ∧ ¬b → post
    # 注意 2 和 3 有隐含的 forall。
    # vcs << inv
    puts "循环不变式是 #{inv}"
   # vcs << [:and]
    vcs << [:implies, [:and, inv, b], wp(body, inv, vcs)]
    vcs << [:implies, [:and, inv, [:not, b]], post]
    
    inv
  else
    raise "不支持语句: #{stmt}"
  end
end
 
SXP.write wp(SXP.read("(set x (+ x 1))"), SXP.read("(> x 5)"), [])
SXP.write wp(SXP.read("(seq (set x 1) (set y (+ x 2)))"), SXP.read("(= y 3)"), [])
SXP.write wp(SXP.read("(if (> x 0) (set y x) (set y (- 0 x)))"), SXP.read("(> y 0)"), [])
SXP.write wp(SXP.read("(seq (set x 1) (set y 2) (set z (+ x y)))"), SXP.read("(= z 3)"), [])
puts "==="
stmt4 = SXP.read("(seq (set sum 0) (set i 1) (while (<= i n) (and (= sum (/ (* (- i 1) i) 2)) (<= 1 i)) (seq (set sum (+ sum i)) (set i (+ i 1)))))")
post4 = SXP.read("(= sum (/ (* n (+ n 1)) 2))")
vcs4 = []
SXP.write wp(stmt4, post4, vcs4)
puts "==="
#p vcs4
SXP.write vcs4

# exit

# -----------------------------------------------------------------------------
# 4. 验证条件检查（使用 Z3）
# -----------------------------------------------------------------------------
def check_valid(formula)
  solver = Z3::Solver.new
  SXP.write(formula)
  #p [:check_valid, formula,sxp_to_z3(formula)]
  solver.assert(p !sxp_to_z3(formula))
  solver.check
  solver.satisfiable? and p solver.model
  !solver.satisfiable?
end
# -----------------------------------------------------------------------------
# 5. 整体验证入口
# -----------------------------------------------------------------------------
def verify(pre_sxp, prog_sxp, post_sxp)
  puts "=" * 60
  puts "📌 前置条件 P: #{pre_sxp}"
  puts "📌 程序: #{prog_sxp}"
  puts "📌 后置条件 Q: #{post_sxp}"
  puts "=" * 60

  vcs = []
  wp = wp(prog_sxp, post_sxp, vcs)
  vcs.unshift [:implies, pre_sxp, wp]
  
  puts "\n🔍 生成验证条件："
  all_ok = true
  
  SXP.write(vcs)
  vcs.each_with_index do |vc, i|
    puts "VC#{i+1}:"
    if check_valid(vc)
      puts "✅ 成立"
    else
      puts "❌ 不成立"
      all_ok = false
    end
  end

  puts "\n" + (all_ok ? "🎉 验证通过！" : "⚠️ 验证失败！")
end

 

# -----------------------------------------------------------------------------
# 6. 从 SXP 字符串解析 + 示例（while 求和）
# -----------------------------------------------------------------------------
if __FILE__ == $0
  # SXP 字符串：
  # 计算 1+2+...+n，i从1到n，sum累加
  sxp_str = <<~SXP
    (seq
      (set sum 0)
      (set i 1)
      (while (<= i n)
        (and (= sum (/ (* (- i 1) i) 2)) (<= 1 i) (<= i (+ n 1)))
        (seq
          (set sum (+ sum i))
          (set i (+ i 1))
        )
      )
    )
  SXP

  prog = SXP.read(sxp_str)
  pre = SXP.read("(< 1 n)") # [:<=, 1, :n]
#  post = [:==, :sum, [:/, [:* ,:n, [:+, :n, 1]], 2]]
  post = SXP.read("(== sum (/ (* n (+ n 1)) 2))")
  verify(pre, prog, post)
end
