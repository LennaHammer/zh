# ==============================================================================
# 完整版 Datalog 解释器
# 支持：标准语法 + 分层否定 + 相等/不等 + 分层检查 + 美化查询输出
# ==============================================================================

# ---------------
# 1. 词法分析
# ---------------
def tokenize(s)
    s
    .gsub("(", " ( ")
    .gsub(")", " ) ")
    .gsub(".", " . ")
    .gsub(",", " , ")
    .gsub(":-", " :- ")
    .gsub("\\+", " \\not ")
    .gsub("!=", " != ")
    .gsub("=", " = ")
    .gsub(" ! = "," != ")
    .split(/\s+/)
end

# ---------------
# 2. 语法解析
# ---------------
def parse(tokens)
    p tokens
  $ptr = 0
  facts = []
  rules = []
  queries = []

  while $ptr < tokens.size
    case tokens[$ptr]
    when "?-"
      $ptr += 1
      atom = parse_atom(tokens)
      queries << atom
      $ptr += 1
      $ptr += 1 if tokens[$ptr] == "."
    else
      head = parse_atom(tokens)
      $ptr += 1
      if tokens[$ptr] == ":-"
        $ptr += 1
        body = []
        loop do
          case tokens[$ptr]
          when "\\not"
            $ptr += 1
            a = parse_atom(tokens)
            body << [:not, a]
          when "="
            a = tokens[ptr-1].to_sym
            b = tokens[ptr+1].to_sym
            body << [:eq, a, b]
            $ptr += 1
          when "!="
            fail
            a = tokens[ptr-1].to_sym
            b = tokens[ptr+1].to_sym
            body << [:neq, a, b]
            $ptr += 1
          else
            a = parse_atom(tokens)
            body << a
          end
          $ptr += 1
          break if tokens[$ptr] == "."
          $ptr += 1 if tokens[$ptr] == ","
        end
        $ptr += 1
        rules << { head: head, body: body }
      else
        facts << head
        $ptr += 1
      end
    end
  end

  { facts: facts, rules: rules, queries: queries }
end

def parse_atom(tokens)
  pred = tokens[$ptr].to_sym
  $ptr += 1

  raise "期望左括号 '('，但找到: #{tokens[$ptr]} 在 #{tokens[..$ptr]}" unless tokens[$ptr] == '('
  $ptr += 1

  args = []
  while tokens[$ptr] != ')' 
    args << tokens[$ptr].to_sym
    p args: args 
    $ptr += 1
    $ptr += 1 if tokens[$ptr] == ','
  end
  fail if tokens[$ptr] != ')'
  # $ptr += 1
  [pred] + args
end

# ---------------
# 3. 辅助函数
# ---------------
def var?(sym)
  sym =~ /^[A-Z]/
end

def substitute(term, subst)
  var?(term) ? subst.fetch(term, term) : term
end

def match(atom, tuple)
  return nil if atom[0] != tuple[0]
  subst = {}
  atom[1..].zip(tuple[1..]) do |a, b|
    if var?(a)
      subst[a] = b unless subst.key?(a)
      return nil if subst[a] != b
    else
      return nil if a != b
    end
  end
  subst
end

# ---------------
# 4. 主体求解（支持等/不等/否定）
# ---------------
def solve_body(body, facts)
  return [{}] if body.empty?
  first, *rest = body
  subs_rest = solve_body(rest, facts)

  case first
  in [:not, atom]
    subs_rest.filter do |s|
      inst = atom.map { |x| substitute(x, s) }
      facts.none? { |t| match(inst, t) }
    end
  in [:eq, a, b]
    subs_rest.filter do |s|
      va = substitute(a, s)
      vb = substitute(b, s)
      va == vb
    end
  in [:neq, a, b]
    subs_rest.filter do |s|
      va = substitute(a, s)
      vb = substitute(b, s)
      va != vb
    end
  else
    res = []
    facts.each do |t|
      subs_rest.each do |s|
        inst = first.map { |x| substitute(x, s) }
        if m = match(inst, t)
          res << s.merge(m)
        end
      end
    end
    res.uniq
  end
end

# ---------------
# 5. 分层（Strata）计算，保证否定安全
# ---------------
def strata(rules)
  dep = Hash.new { |h,k| h[k] = Set.new }
  strat = Hash.new(0)

  rules.each do |r|
    head = r[:head][0]
    r[:body].each do |lit|
      case lit
      in [:not, a]
        dep[head] << a[0]
        strat[head] = [strat[head], strat[a[0]] + 1].max
      in Array
        # 忽略等/不等
      else
        dep[head] << lit[0]
        strat[head] = [strat[head], strat[lit[0]]].max
      end
    end
  end

  strat
end

# ---------------
# 6. 分层不动点求值
# ---------------
def stratified_eval(facts, rules)
  strat = strata(rules)
  layers = strat.group_by { |k,v| v }.sort.map { |_,vs| vs.map(&:first) }
  current = facts.dup

  layers.each do |preds|
    loop do
      new_facts = []
      rules.each do |r|
        next unless preds.include?(r[:head][0])
        subs = solve_body(r[:body], current)
        subs.each do |s|
          head = r[:head].map { |x| substitute(x, s) }
          new_facts << head unless current.include?(head)
        end
      end
      break if new_facts.empty?
      current += new_facts
    end
  end

  current.uniq
end

# ---------------
# 7. 查询与美化输出
# ---------------
def run_query(goal, facts)
  res = []
  facts.each do |t|
    if s = match(goal, t)
      res << s
    end
  end
  res.uniq
end

def pretty_answer(subst)
  return "yes" if subst.empty?
  subst.map { |k,v| "#{k} = #{v}" }.join(", ")
end

# ---------------
# 8. 总入口
# ---------------
def run_datalog(source)
  puts "=" * 50
  puts "Datalog 程序"
  puts "=" * 50
  puts source

  ast = parse(tokenize(source))
  db = stratified_eval(ast[:facts], ast[:rules])

  puts "\n" + "=" * 50
  puts "查询结果"
  puts "=" * 50

  ast[:queries].each do |q|
    goal_str = "#{q[0]}(#{q[1..].join(', ')})"
    puts "?- #{goal_str}."
    answers = run_query(q, db)

    if answers.empty?
      puts " no."
    else
      answers.each do |ans|
        puts " #{pretty_answer(ans)}"
      end
    end
    puts
  end
end

# ==============================================================================
# 示例测试（包含：递归、否定、相等、不等）
# ==============================================================================

code = <<~DAT
parent(alice, bob).
parent(bob, charlie).
parent(charlie, david).
parent(eve, frank).

ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

childless(X) :- \\+ parent(X, Y).
different(X, Y) :- X != Y.

?- ancestor(alice, Y).
?- childless(X).
?- different(alice, bob).
DAT

run_datalog(code)