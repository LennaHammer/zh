# 2024-12-27-用Python分析程序代码.md

```python
# Weakest Precondition
# 背景知识：Hoare Logic
# 这里最弱的意思是能从 require -> wp，比如wp要求是正数，而实际输入是大于10，即输入是wp的子集，wp的范围更大限制更松。
# wp可以针对不同语句根据规则生成
# 特别注意循环语句，根据所给的某一个正确的循环不变量来产生 wp。
# 考虑5种语句
# 1. 空语句 wp(SKIP,Q)=Q
# 2. 序列语句  wp(S1;S2, Q) = wp(S1, wp(S2,Q))
# 3. 赋值语句 wp(x=e, Q) = Q[e/x]
# 4. 条件语句 wp(IF B THEN S1 ELSE S2, Q) = ( B => wp(S1,Q) Λ !B => wp(S2,Q)) = (B Λ wp(S1, Q)) V (!B Λ wp(S2, Q))
# 5. 循环语句 wp(WHILE B DO S DONE)=Inv
#     - 额外约束 forall  1. Inv/\B->wp(S,I) 2. Inv/\~B->Q
# 在验证中还涉及
# 1. 断言 wp(assert(B),Q)=B/\Q
# 2. 假设 wp(assume(B),Q)=B->Q
# 参考
# - https://www.cs.williams.edu/~freund/cs326/ReasoningPart1.html#weakest-preconditions
# - https://www.philipzucker.com/weakest-precondition-z3py/
#
import z3.z3util
import operator
import sexpdata
from sexpdata import Symbol
import z3
x = sexpdata.loads('(begin (set x (+ x 1)))')

print(x)
bin_ops = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "=": operator.eq,
    'and': z3.And,
}


def make_expr(x, env):
    if isinstance(x, list):
        op = str(x[0])
        if op in bin_ops.keys():
            return bin_ops[op](make_expr(x[1], env), make_expr(x[2], env))
        assert False, x
    elif isinstance(x, Symbol):
        return env[str(x)]
    elif isinstance(x, int):
        x = z3.IntSort().cast(x)
        return x
    else:
        raise NotImplementedError


print(repr(make_expr(sexpdata.loads("(+ x 2)"), {"x": z3.Int("x")})))
# exit()


def wp(s, q, env):
    print('wp', s, q, env)
    if isinstance(s, list):
        op = str(s[0])
        if op == 'begin':
            p = q
            for x in s[-1:0:-1]:
                p = wp(x, p, env)
        elif op == 'set':
            print("set", [(env[str(s[1])], make_expr(s[2], env))])
            p = z3.substitute(q, [(env[str(s[1])], make_expr(s[2], env))])
        elif op == 'if':
            assert 3 <= len(s) <= 4
            e = make_expr(s[1], env)
            p = z3.Implies(e, wp(s[2], q, env))
            if len(s) == 4:
                p = z3.And(p, z3.Implies(z3.Not(e), wp(s[3], q, env)))
        elif op == 'while':
            b = make_expr(s[1], env)
            assert s[2][0] == sexpdata.Symbol("invariant")
            I = make_expr(s[2][1], env)
            vs = z3.z3util.get_vars(q)
            p = z3.And(
                I,
                z3.ForAll(vs, z3.And(
                    z3.Implies(z3.And(I, b), wp(s[3], I, env)),
                    z3.Implies(z3.And(I, z3.Not(b)), q)
                ))
            )
        else:
            raise NotImplementedError("%s" % op)
        return p
    assert None, s


def verify(code, vars, requires, ensures):
    u = z3.Bool("u")
    code = sexpdata.loads(code)
    print(code)
    p = wp(code, ensures, vars)
    print(p)
    cond = z3.Implies(requires, p)
    solver = z3.Solver()
    solver.add(z3.Not(cond))
    print(solver)
    sat = solver.check()
    print('SAT>', (sat))
    # print(solver.check())
    if sat == z3.sat:
        print("不成立")
        print(solver.model())
    elif sat == z3.unsat:
        print("代码成立")
    return sat == z3.unsat


x, y, z = z3.Int("x"), z3.Int("y"), z3.Int('z')

verify(
    "(set x (+ x 2))",
    vars={"x": x},
    requires=x == 0,
    ensures=x <= 2,
)
verify(
    "(begin (set x (+ x 2)) (set x (+ x 3)) )",
    vars={"x": x, 'y': y},
    requires=x == y,
    ensures=x == y+5,
)
verify(
    "(begin (if (< x y) (set z y) (set z x)))",
    vars={"x": x, 'y': y, 'z': z},
    requires=True,
    ensures=z3.And(z >= x, z >= y),
)
verify(
    """
    (begin 
      (set x 0)
      (while (< x 7)
        (invariant (<= x 7))
        (set x (+ x 1))
        ))""",
    vars={"x": x},
    requires=True,
    ensures=x == 7,
)
# exit()
# 失败的例子
verify(
    "(begin (if (< x 1) (set x (+ x 2)) (set x (+ x 3))))",
    vars={"x": x},
    requires=x == 0,
    ensures=x == 3,
)
```




# 2024-12-28-想写一个小说.md


表达欲。


结局。

构思。

设定。




# 2024-12-29-书架.md

一些书架上的书

[The Project Gutenberg eBook of The Mysterious Affair at Styles, by Agatha Christie](https://www.gutenberg.org/files/863/863-h/863-h.htm)

