# 代码有 bug。
require 'sxp'

$type_variable_counter = 0

def new_type_variable 
  :"t#{$type_variable_counter+=1}"
end

def free_vars(type)
  case type
  in :int | :bool then []
  in Symbol then [type] #  if type=~/^t\d+$/
  in [:'->', param_type, return_type] then free_vars(param_type) | free_vars(return_type)
  else []
  end
end

def apply_subst(subst, type)
  case type
  in Symbol then subst.fetch(type, type)
  in [:'->', param_type, return_type] then [:'->', apply_subst(subst, param_type), apply_subst(subst, return_type)]
  else type
  end
end

def unify(type1, type2, subst)
  type1_applied = apply_subst(subst, type1)
  type2_applied = apply_subst(subst, type2)
  
  return subst if type1_applied == type2_applied
  p [:test, type1_applied, type2_applied]
  case [type1_applied, type2_applied]
  in [Symbol => type_var, other_type] if type_var=~/^t\d+$/
    raise "循环类型" if free_vars(other_type).include?(type_var)
    subst.merge(type_var => other_type)
  in [other_type, Symbol => type_var] if type_var=~/^t\d+$/
    unify(type_var, other_type, subst)
  in [[:'->', *xs], [:'->',*ys]]
    xs.zip(ys).each do |param1, param2|
      subst = unify(param1, param2, subst)
    end
    subst
  # in [[:'->', param1, ret1], [:'->', param2, ret2]]
  #   subst1 = unify(param1, param2, subst)
  #   ret1_applied = apply_subst(subst1, ret1)
  #   ret2_applied = apply_subst(subst1, ret2)
  #   unify(ret1_applied, ret2_applied, subst1)
  else
    p [type1, type2]
    raise "unification fail: #{type1_applied} != #{type2_applied}"
  end
end

def generalize(type_env, type)
  env_free_vars = type_env.values.flat_map { |schema| free_vars(schema[2]) }.uniq
  [:forall, free_vars(type) - env_free_vars, type]
end

def instantiate(schema)
  case schema
  in [:forall, type_vars, inner_type]
    p schema
    subst = type_vars.to_h { |var| [var, new_type_variable] }
    apply_subst(subst, inner_type)
  else
    schema
  end
end

def apply_subst_to_env(subst, type_env)
  type_env.transform_values { |schema|
  if schema.is_a?(Array) && schema.first == :forall
    [:forall, schema[1], apply_subst(subst, schema[2])]
  else
    schema
  end 
  }
end

def type_of(expression, type_env, subst)
  case expression
  in Integer
    [:int, subst]
  in true | false | :true | :false
    [:bool, subst]
  in Symbol
    p env: type_env.keys, expr: expression
    [instantiate(type_env.fetch(expression)), subst]
  in [:begin, *expressions]
    type = nil
    expressions.each do |expr|
      type, subst = type_of(expr, type_env, subst)
    end
    [type, subst]
  in [:lambda, [param], body]
    fresh_var = new_type_variable
    body_type, subst1 = type_of(body, type_env.merge(param => fresh_var), subst)
    [[:'->', apply_subst(subst1, fresh_var), body_type], subst1]
  in [func_expr, *xs] if func_expr!=:let # apply
    p func_expr: func_expr, subst: subst, env: type_env
    func_type, subst = type_of(func_expr, type_env, subst)
    p func_type: func_type, subst: subst
    arg_types = xs.map{|e|
      e_type, subst = type_of(e, apply_subst_to_env(subst, type_env), subst)
      e_type
    }
#    arg_type, subst = type_of(arg_expr, apply_subst_to_env(subst, type_env), subst)
    func_type_applied = apply_subst(subst, func_type) # 这行需要么？
    p func_type_applied: func_type_applied, subst: subst
    return_type_var = new_type_variable
    subst = unify(func_type_applied, [:'->', *arg_types, return_type_var], subst)
    [apply_subst(subst, return_type_var), subst]
  in [:let, var_name, expr1, expr2]
    type1, subst1 = type_of(expr1, type_env, subst)
    env1 = apply_subst_to_env(subst1, type_env)
    schema = generalize(env1, type1)
    type2, subst2 = type_of(expr2, env1.merge(var_name => schema), subst1)
    [type2, subst2]
  end
end

def run(code)
  expr = SXP.read(code)
  env = {
    :true => :bool, 
    :false => :bool,
    :+ => [:'->', :int, :int, :int],
  }
  type, _ = type_of(expr, env, {})
  puts "🎉type: #{type}"
  puts
end

# ------------------------------------------------------------------------------
# 测试
# ------------------------------------------------------------------------------
run "42" # 整数类型
run "true" # 布尔类型
run "(+ 1 2)"
run "(lambda (x) x)" # lambda
run "((lambda (x) x) 42)" # apply
run "(let id (lambda (x) x) (begin (id 42) (id true)))" # let 多态
run "(lambda (id) (begin (id 42) (id true))) (lambda (x) x))" # 故意报错



# run "(let id (lambda (x) x) (let f (lambda (y) (id y)) f))"
# run "(let id (lambda (x) x) (id true))"
# run "(lambda (x) (lambda (y) x))"
# run "((lambda (x) (lambda (y) x)) 42)"
# run "(let k (lambda (x) (lambda (y) x)) ((k 1) 2))"
# run "(lambda (f) (lambda (g) (lambda (x) (f (g x)))))"
# run "(let id (lambda (x) x) (let const (lambda (y) (lambda (z) y)) (const id)))"
# run "(let x 1 (let y 2 (let z 3 x)))"
# run "(let b true (let f (lambda (x) b) (f false)))"

    