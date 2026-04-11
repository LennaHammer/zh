import sexpdata
import z3  # 显式导入Z3模块，无全局导入

# S表达式 → Z3表达式 转换（全显式z3调用）
def to_z3(e, env):
    if isinstance(e, sexpdata.Symbol):
        return z3.Int(str(e))
        v = str(e)
        env[v] = z3.Int(v) if v not in env else env[v]
        return env[v]
    if isinstance(e, int):
        return z3.IntVal(e)
    op, *args = e
    a1, a2 = to_z3(args[0], env), to_z3(args[1], env)
    return {"+":a1+a2, "-":a1-a2, "<":a1<a2, "=":a1==a2, ">=":a1>=a2, "<=":a1<=a2,"<":a1<a2,">":a1>a2}[str(op)]
    # '->':z3.Implies(a1,a2)

# 最弱前置条件(WP)递归计算
def wp(cmd, post, env):
    head = str(cmd[0])
    if head == "set": # 赋值: WP(x=e, Q) = Q[x→e]
        return z3.substitute(post, (to_z3(cmd[1], env), to_z3(cmd[2], env)))
    if head == "seq": # 顺序: WP(c1;c2, Q) = WP(c1, WP(c2, Q))
        return wp(cmd[1], wp(cmd[2], post, env), env)
    if head == "if": # 条件: WP(if b t e, Q) = (b→WP(t,Q)) ∧ (¬b→WP(e,Q))
        c = to_z3(cmd[1], env)
        return z3.Implies(c, z3.And(wp(cmd[2], post, env)), z3.Implies(z3.Not(c), wp(cmd[3], post, env)))
    if head == "while": # 循环: 用不变式I，WP(while,Q)=I 且 
        return to_z3(cmd[2], env)
    return post

# 验证入口
def verify(sexp_code, pre='true', post='false'):
    prog = sexpdata.loads(sexp_code)
    env = {}
    post_z3 = to_z3(sexpdata.loads(post), env)
    pre_z3 = wp(prog, post_z3, env)
    s = z3.Solver()

    # 循环验证（不变式从while语句内嵌提取）
    if str(prog[0]) == "while":
        cond_z3 = to_z3(prog[1], env)
        inv_z3 = pre_z3
        induct = z3.Implies(z3.And(inv_z3, cond_z3), wp(prog[3], inv_z3, env))
        post_cond = z3.Implies(z3.And(inv_z3,z3.Not(cond_z3)), post_z3)
        s.add(z3.Not(z3.And(induct, post_cond))) # Z3 中自由变量隐式全称量化么？不能。
        # vs = z3.z3util.get_env(q)
        # 用 forall 或者单独的 solver。
        # vc = z3.ForAll(list(env.values()), z3.And(induct, post_cond))
        # s.add(z3.Not(vc))
    else:
        s.add(z3.Not(pre_z3))

    # 输出结果（显式判断z3.unsat）
    res = s.check()
    print(f"验证结果: {res}")
    print(f"结论: {'✅程序正确' if res == z3.unsat else '❌存在反例'}")
    if res == z3.sat:
        print(f"反例: {s.model()}")
    print("-"*50)

# ------------------- 演示示例 -------------------
if __name__ == "__main__":
    print("【赋值】x = x+1，验证 x>5")
    verify("(set x (+ x 1))", post="(> x 5)")

    # 循环：不变式直接写在while语句中
    print("【循环】while x<10: x=x+1，不变式 x≤10，验证 x=10")
    verify("(while (< x 10) (<= x 10) (set x (+ x 1)))", post="(= x 10)")

    print("【条件】if x<0: x=-x，验证 x≥0")
    verify("(if (< x 0) (set x (- 0 x)) (set x x))", post="(>= x 0)")