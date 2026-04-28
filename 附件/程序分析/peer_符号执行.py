"""
A Peer Architecture for Lightweight Symbolic Execution
======================================================
论文: Bruni, Disney, Flanagan — UC Santa Cruz

严格按论文 Figure 4/5/6 的伪代码复现。
核心思路：不写新解释器/编译器，而是利用 Python 操作符动态分发（__add__, __bool__ 等），
让符号执行引擎作为"对等体"（peer）与目标程序运行在同一进程中。

架构层次（对应论文 Figure 1）：
  ┌─────────────────────────────────┐
  │  Target Program                 │
  │  (def abs(x): if x>=0: ...)    │
  └──────────┬──────────────────────┘
             │ 操作符回调 (__add__, __bool__, ...)
  ┌──────────▼──────────────────────┐
  │  Proxy Objects                  │
  │  IntProxy / BoolProxy           │
  └──────────┬──────────────────────┘
             │ SMT 接口
  ┌──────────▼──────────────────────┐
  │  Z3 SMT Solver                  │
  └─────────────────────────────────┘
"""

from z3 import *

# ====================================================================
# 第 1 层：SMT 求解器接口（论文 Figure 4）
# ====================================================================
# 论文中的 smt_* 函数直接映射为 Z3 操作：
#
#   smt_mkvar()              → Int(name)       创建符号变量
#   smt_mkint(n)             → IntVal(n)        创建常量
#   smt_op('+', a, b)        → a + b            算术运算（Z3 重载了 +）
#   smt_op('-', a, b)        → a - b
#   smt_op('*', a, b)        → a * b
#   smt_pred('=', a, b)      → a == b           谓词
#   smt_pred('<', a, b)      → a < b
#   smt_pred('!', f)         → Not(f)           公式取反
#   smt_fop('&', [f1,f2])   → And(f1, f2)      公式合取
#   smt_solve(formula)       → Solver().check()  可满足性检查
#
# 下面封装成与论文一致的函数签名：

_var_counter = 0

def smt_mkvar():
    """Figure 4: smt_mkvar : Unit → Term"""
    global _var_counter
    name = f"x{_var_counter}"
    _var_counter += 1
    return Int(name)

def smt_mkint(n: int):
    """Figure 4: smt_mkint : Int → Term"""
    return IntVal(n)

def smt_op(op: str, *terms):
    """Figure 4: smt_op : String × Term* → Term"""
    if op == '+':    return terms[0] + terms[1]
    if op == '-':    return terms[0] - terms[1]
    if op == '*':    return terms[0] * terms[1]
    if op == '/':    return terms[0] / terms[1]
    raise ValueError(f"Unknown op: {op}")

def smt_pred(pred: str, *args):
    """Figure 4: smt_pred : String × Term* → Formula"""
    if pred == '=':   return args[0] == args[1]
    if pred == '<':   return args[0] < args[1]
    if pred == '>':   return args[0] > args[1]
    if pred == '<=':  return args[0] <= args[1]
    if pred == '>=':  return args[0] >= args[1]
    if pred == '!':   return Not(args[0])
    raise ValueError(f"Unknown pred: {pred}")

def smt_fop(op: str, *formulas):
    """Figure 4: smt_fop : String × Formula* → Formula"""
    if op == '&':   return And(*formulas)
    if op == '|':   return Or(*formulas)
    if op == '!':   return Not(formulas[0])
    raise ValueError(f"Unknown fop: {op}")

def smt_solve(formula) -> bool:
    """Figure 4: smt_solve : Formula → Bool"""
    s = Solver()
    s.add(formula)
    return s.check() == sat


# ====================================================================
# 第 2 层：IntProxy —— 符号整数代理（论文 Figure 5）
# ====================================================================

class IntProxy:
    """
    Figure 5: Symbolic Execution Engine — Integer Proxies

    对等体（Peer）：Python int 的符号版本。
    每个算术/比较运算返回新的 Proxy，构建出 SMT 表达式树。
    不持有具体值 —— 全部符号化。
    """

    def __init__(self, term):
        self.term = term  # Z3 ArithRef（符号项）

    # ---- 算术运算（Figure 5 lines 1-11） ----
    def __add__(self, other):
        """Figure 5 line 1-6"""
        if type(other) == int:
            other = IntProxy(smt_mkint(other))
        return IntProxy(smt_op('+', self.term, other.term))

    def __radd__(self, other):
        """Figure 5 line 8-11"""
        return self.__add__(other)

    def __sub__(self, other):
        if type(other) == int:
            other = IntProxy(smt_mkint(other))
        return IntProxy(smt_op('-', self.term, other.term))

    def __rsub__(self, other):
        if type(other) == int:
            other = IntProxy(smt_mkint(other))
        return IntProxy(smt_op('-', other.term, self.term))

    def __mul__(self, other):
        if type(other) == int:
            other = IntProxy(smt_mkint(other))
        return IntProxy(smt_op('*', self.term, other.term))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if type(other) == int:
            other = IntProxy(smt_mkint(other))
        return IntProxy(smt_op('/', self.term, other.term))

    def __floordiv__(self, other):
        if type(other) == int:
            other = IntProxy(smt_mkint(other))
        return IntProxy(smt_op('/', self.term, other.term))

    def __rfloordiv__(self, other):
        if type(other) == int:
            other = IntProxy(smt_mkint(other))
        return IntProxy(smt_op('/', other.term, self.term))

    def __neg__(self):
        return IntProxy(smt_op('*', smt_mkint(-1), self.term))

    # ---- 比较运算（Figure 5 lines 13-23） ----
    def __eq__(self, other):
        """Figure 5 line 13-18"""
        if type(other) == int:
            other = IntProxy(smt_mkint(other))
        return BoolProxy(smt_pred('=', self.term, other.term))

    def __req__(self, other):
        """Figure 5 line 20-23"""
        return self.__eq__(other)

    def __ne__(self, other):
        if type(other) == int:
            other = IntProxy(smt_mkint(other))
        return BoolProxy(smt_pred('!', smt_pred('=', self.term, other.term)))

    def __lt__(self, other):
        if type(other) == int:
            other = IntProxy(smt_mkint(other))
        return BoolProxy(smt_pred('<', self.term, other.term))

    def __le__(self, other):
        if type(other) == int:
            other = IntProxy(smt_mkint(other))
        return BoolProxy(smt_pred('<=', self.term, other.term))

    def __gt__(self, other):
        if type(other) == int:
            other = IntProxy(smt_mkint(other))
        return BoolProxy(smt_pred('>', self.term, other.term))

    def __ge__(self, other):
        if type(other) == int:
            other = IntProxy(smt_mkint(other))
        return BoolProxy(smt_pred('>=', self.term, other.term))

    def __repr__(self):
        return f"IntProxy({self.term})"


# ====================================================================
# 第 3 层：BoolProxy + 模型检查循环（论文 Figure 6）
# ====================================================================

# 论文 Figure 6 lines 1-2: 全局路径状态
__path__ = []           # 当前路径中每个自由分支的选择（True/False）
__pathcondition__ = []  # 当前路径的约束公式列表
_in_contract = False     # 契约检查期间不修改 __path__（避免 helper 函数干扰路径）

MAX_DEPTH = 10          # 论文 line 48: 自由分支深度限制（SmallCheck 风格）


class DepthException(Exception):
    """论文 Figure 6 line 4"""
    pass


class BoolProxy:
    """
    Figure 6 lines 24-53: Boolean Proxies

    __bool__ 的 5 步决策算法（对应论文 5.4 节）：
    1. 若仅 true 分支可满足  → 强制走 true（不计入深度）
    2. 若仅 false 分支可满足 → 强制走 false（不计入深度）
    3. 若 __path__ 中已预定  → 按预定路径走
    4. 若深度超限           → 抛出 DepthException，剪枝
    5. 默认                 → 走 true，记录为自由分支
    """

    def __init__(self, formula):
        """Figure 6 line 25"""
        self.formula = formula  # Z3 BoolRef（符号公式）

    def __not__(self):
        """Figure 6 line 27"""
        return BoolProxy(smt_fop('!', self.formula))

    def __bool__(self):
        """
        Figure 6 lines 29-53: 论文 5.4 节的核心决策算法
        """
        global __path__, __pathcondition__, _in_contract

        pc = And(*__pathcondition__) if __pathcondition__ else True

        # Step 1 & 2: 检查两个分支的可满足性（论文 lines 32-33）
        true_cond  = smt_solve(smt_fop('&', pc, self.formula))
        false_cond = smt_solve(smt_fop('&', pc, smt_pred('!', self.formula)))

        # 强制分支：仅一侧可行 → 不计入路径深度（论文 lines 37-38）
        if true_cond and not false_cond:
            return True
        if false_cond and not true_cond:
            return False

        # 契约检查模式：只做可行性判断，不修改 __path__
        if _in_contract:
            return True  # 两个都可行 → 条件可能成立

        # Step 3: 按预定路径走（论文 lines 40-45）
        if len(__path__) > len(__pathcondition__):
            branch = __path__[len(__pathcondition__)]
            __pathcondition__.append(
                self.formula if branch else smt_pred('!', self.formula)
            )
            return branch

        # Step 4: 深度限制，剪枝（论文 line 48）
        if len(__path__) >= MAX_DEPTH:
            raise DepthException(f'Depth {MAX_DEPTH} exceeded')

        # Step 5: 默认走 true，记录为新的自由分支（论文 lines 51-53）
        __path__.append(True)
        __pathcondition__.append(self.formula)
        return True

    def __repr__(self):
        return f"BoolProxy({self.formula})"


# ====================================================================
# 第 4 层：test() 模型检查主循环（论文 Figure 6 lines 6-22）
# ====================================================================

def test(f, *args, **kwargs):
    """
    Figure 6 lines 6-22: The Model Checking Loop

    符号执行的主入口。反复调用被测函数 f，
    每次执行后回溯 __path__ 以探索新分支。

    回溯策略（论文 lines 16-22）：
    1. 弹出所有已被完全探索的 False 分支
    2. 若 __path__ 为空 → 整棵分支树已穷举，结束
    3. 将最后一个 True 翻转为 False → 下一次执行探索另一侧
    """
    global __path__, __pathcondition__

    __path__ = []               # 论文 line 8: 初始化路径
    iteration = 0

    while True:                 # 论文 line 9
        __pathcondition__ = []  # 论文 line 10: 重置路径条件
        iteration += 1

        try:
            result = f(*args, **kwargs)         # 论文 line 13
            print(f"[iter {iteration}] path={__path__}  result={result}")
        except DepthException:
            print(f"[iter {iteration}] path={__path__}  PRUNED (depth limit)")
        except ContractViolation as e:
            # 论文 Section 6: pre/arg 失败 → 忽略；post/inv 失败 → bug
            if e.kind in ('pre', 'arg'):
                print(f"[iter {iteration}] path={__path__}  IGNORED ({e.kind} violation: 非bug)")
            else:
                print(f"[iter {iteration}] path={__path__}  BUG: {e}")
        except Exception as e:
            print(f"[iter {iteration}] path={__path__}  EXCEPTION: {e}")

        # ---- 论文 lines 16-22: 选择下一条探索路径 ----
        # 1. 弹出已完全探索的 False 分支（论文 line 18）
        while len(__path__) > 0 and not __path__[-1]:
            __path__.pop()

        # 2. 如果路径为空，整棵分支树已穷举（论文 line 20）
        if __path__ == []:
            print(f"\n全部 {iteration} 条路径探索完毕。")
            return

        # 3. 翻转最后一个 True 为 False（论文 line 22）
        __path__[-1] = False

    return result


# ====================================================================
# 第 5 层：契约系统（论文 Section 6）
# ====================================================================
# 论文 Section 6 描述了约 100 行的契约库，使用 Python 装饰器语法：
#
#   @inv(condition)    — 类不变量，在 __init__ 后和每个方法调用后检查
#   @pre(condition)    — 前置条件，方法调用前检查
#   @post(condition)   — 后置条件，方法调用后检查（别名 @ensure）
#   @arg(condition)    — 参数条件
#
# 论文的 bug 判定规则：
#   - @pre / @arg 失败 → 忽略（输入不合法，不是 bug）
#   - @post / @inv 失败 → 报告为 bug
#
# 当后置条件失败时，论文以 concolic 方式从 SMT 求解器取回具体值，
# 用真实值重新执行以展示 bug 的实际行为。


def _is_proxy(obj):
    """检查对象是否为 Proxy 类型"""
    return isinstance(obj, (IntProxy, BoolProxy))


def _check_condition(cond, args, kwargs, result=None, contract_mode=True):
    """
    评估契约条件。

    contract_mode=True (post/inv):
      - 设置 _in_contract 防止污染 __path__
      - 对 BoolProxy 用 Z3 检查 PC → condition

    contract_mode=False (pre/arg):
      - 正常执行，让条件中的分支参与路径探索
      - BoolProxy.__bool__ 自行处理可行性
    """
    global _in_contract
    if contract_mode:
        _in_contract = True
    try:
        import inspect
        sig = inspect.signature(cond)
        try:
            ok = cond(*args, **kwargs, result=result)
        except TypeError:
            ok = cond(*args, **kwargs)
    except Exception:
        ok = cond(*args, **kwargs)
    finally:
        if contract_mode:
            _in_contract = False

    if isinstance(ok, BoolProxy):
        if contract_mode:
            return _check_proxy_condition(ok)
        else:
            # 让 BoolProxy.__bool__ 自然决定（已在上面触发）
            return bool(ok)
    return ok


def _check_proxy_condition(bp: BoolProxy):
    """
    检查 BoolProxy 条件在当前路径下是否成立。

    契约条件应该对所有可达状态成立。
    若 PC ∧ ¬condition 可满足，说明存在反例 → 条件失败。
    若 PC ∧ ¬condition 不可满足（即 PC → condition 永真），条件成立。
    """
    pc = And(*__pathcondition__) if __pathcondition__ else True
    s = Solver()
    s.add(pc)
    s.add(smt_pred('!', bp.formula))  # 尝试让条件为假
    if s.check() == sat:
        return False  # 存在反例 → 契约违反
    return True  # 条件在所有可达状态下成立


def _extract_counterexample():
    """从当前路径条件中提取具体反例（论文 Section 7 的 concolic 回溯）"""
    if not __pathcondition__:
        return {}
    s = Solver()
    s.add(And(*__pathcondition__))
    if s.check() == sat:
        m = s.model()
        return {d.name(): m[d].as_long() for d in m.decls()}
    return {}


class ContractViolation(Exception):
    """契约违反异常"""
    def __init__(self, kind, message, counterexample=None):
        super().__init__(message)
        self.kind = kind          # 'pre', 'post', 'inv', 'arg'
        self.counterexample = counterexample


def pre(condition):
    """
    论文 Section 6: @pre(condition)

    前置条件。失败时忽略（输入不合法，不是 bug）。
    contract_mode=False: 让条件作为正常分支参与路径探索。
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            ok = _check_condition(condition, args, kwargs, contract_mode=False)
            if not ok:
                raise ContractViolation('pre', f"前置条件失败: {condition.__name__}")
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__wrapped__ = func
        return wrapper
    return decorator


def post(condition):
    """
    论文 Section 6: @post(condition) / @ensure(condition)

    后置条件。失败时报告为 bug。
    contract_mode=True: 用 Z3 直接检查 PC → condition，不污染 __path__。
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            ok = _check_condition(condition, args, kwargs, result=result,
                                  contract_mode=True)
            if not ok:
                ce = _extract_counterexample()
                raise ContractViolation('post',
                    f"后置条件失败: {condition.__name__}  |  反例: {ce}",
                    counterexample=ce)
            return result
        wrapper.__name__ = func.__name__
        wrapper.__wrapped__ = func
        return wrapper
    return decorator


# 别名（论文 Figure 8/9 使用 @ensure）
ensure = post


def arg(condition):
    """
    论文 Section 6: @arg(condition)

    参数条件。失败时忽略（等同于前置条件）。
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            ok = _check_condition(condition, args, kwargs, contract_mode=False)
            if not ok:
                raise ContractViolation('arg', f"参数条件失败: {condition.__name__}")
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__wrapped__ = func
        return wrapper
    return decorator


def inv(condition):
    """
    论文 Section 6: @inv(condition)

    类不变量。在 __init__ 后和每个方法调用后检查。
    失败时报告为 bug。
    """
    def decorator(cls):
        # 包装所有方法（包括 __init__）
        for name, method in list(cls.__dict__.items()):
            if not name.startswith('_') or name == '__init__':
                if callable(method):
                    def make_wrapper(m):
                        def wrapper(self, *args, **kwargs):
                            result = m(self, *args, **kwargs)
                            ok = _check_condition(condition, (self,), {})
                            if not ok:
                                ce = _extract_counterexample()
                                raise ContractViolation('inv',
                                    f"不变量失败: {condition.__name__}  |  反例: {ce}",
                                    counterexample=ce)
                            return result
                        wrapper.__name__ = m.__name__
                        return wrapper
                    setattr(cls, name, make_wrapper(method))
        return cls
    return decorator


# ====================================================================
# 演示：论文中的示例（含契约系统）
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("A Peer Architecture for Lightweight Symbolic Execution")
    print("Bruni, Disney, Flanagan — UC Santa Cruz")
    print("严格按论文 Figure 4/5/6 + Section 6 契约系统 复现")
    print("=" * 60)

    # ================================================================
    # 示例 1 & 2：论文 Figure 2 — abs 和 succ（基础分支）
    # ================================================================
    print("\n" + "-" * 40)
    print("示例 1: abs(x) — 论文 Figure 2")
    print("-" * 40)

    def abs_proxy(x):
        if x >= 0:
            return x
        else:
            return -x

    x = IntProxy(smt_mkvar())
    test(abs_proxy, x)

    print("\n" + "-" * 40)
    print("示例 2: succ(x) — 论文 Figure 2（含 bug: x==42768 触发 fail）")
    print("-" * 40)

    def succ_proxy(x):
        if x == 42768:
            raise RuntimeError("fail()")
        return x + 1

    x = IntProxy(smt_mkvar())
    test(succ_proxy, x)

    # ================================================================
    # 示例 3：论文 Figure 9 — faulty_fact + @ensure 契约
    # ================================================================
    print("\n" + "-" * 40)
    print("示例 3: faulty_fact — 论文 Figure 9（@ensure 契约检测 bug）")
    print("-" * 40)

    def fact_spec(x):
        """论文 Figure 9: 阶乘的参考实现（作为契约的 oracle）"""
        n = 1
        while x > 0:
            n = n * x
            x = x - 1
        return n

    @ensure(lambda x, result=0: fact_spec(x) == result)
    def faulty_fact(x):
        """论文 Figure 9: 含 bug 的阶乘（x==40 返回错误值）"""
        if x == 40:
            return 123456789
        n = 1
        while x > 0:
            n = n * x
            x = x - 1
        return n

    x = IntProxy(smt_mkvar())
    test(faulty_fact, x)

    # ================================================================
    # 示例 4：论文 Figure 8 — QuickSort + @ensure 契约
    # ================================================================
    print("\n" + "-" * 40)
    print("示例 4: quicksort — 论文 Figure 8（@ensure 验证有序性）")
    print("-" * 40)

    def ordered(arr):
        """检查数组是否非严格递增"""
        for i in range(len(arr) - 1):
            if arr[i] > arr[i + 1]:
                return False
        return True

    @ensure(lambda array, result=[]: ordered(result))
    def quicksort(array):
        """论文 Figure 8: 快速排序"""
        if len(array) <= 1:
            return array
        pivot = array[0]
        less, greater = [], []
        for x in array[1:]:
            if x <= pivot:
                less.append(x)
            else:
                greater.append(x)
        return quicksort(less) + [pivot] + quicksort(greater)

    # 用 3 个符号变量测试（论文 7.1 节：6 条路径对应 6 种排列）
    x = IntProxy(smt_mkvar())
    y = IntProxy(smt_mkvar())
    z = IntProxy(smt_mkvar())
    test(quicksort, [x, y, z])

    # ================================================================
    # 示例 5：@pre + @post 契约演示
    # ================================================================
    print("\n" + "-" * 40)
    print("示例 5: @pre + @post — abs 函数的完整契约")
    print("-" * 40)

    @pre(lambda x: x < 0)                            # 前置：只接受负数
    @post(lambda x, result=0: result >= 0)            # 后置：结果非负
    @post(lambda x, result=0: result == -x)           # 后置：结果 == -x
    def abs_neg(x):
        """只对负数求绝对值"""
        return -x

    x = IntProxy(smt_mkvar())
    test(abs_neg, x)

    # ---- 除法：前置条件分支探索 ----
    print("\n" + "-" * 40)
    print("示例 5b: @pre — 前置条件作为路径分支 (b!=0 / b==0)")
    print("-" * 40)

    @pre(lambda a, b: b != 0)
    def divide(a, b):
        return a // b

    a = IntProxy(smt_mkvar())
    b = IntProxy(smt_mkvar())
    test(divide, a, b)

    # ================================================================
    # 示例 6：论文 Figure 12 — 性能测试（符号执行开销极低）
    # ================================================================
    print("\n" + "-" * 40)
    print("示例 6: 松耦合函数的符号执行开销 — 论文 Figure 12 风格")
    print("-" * 40)

    def loose_dependency(x):
        """大量计算与参数 x 无关 → 符号执行几乎无开销（论文 7.4 节）"""
        s = 0
        for i in range(1000):      # 论文 Figure 12 是 1,000,000
            s += i * i
        if x == 42:
            return "Good guess"
        else:
            return "Nope"

    x = IntProxy(smt_mkvar())
    test(loose_dependency, x)

    # ================================================================
    # 示例 7：Peer 对象的透明性（IntProxy 如何构建 SMT 表达式）
    # ================================================================
    print("\n" + "-" * 40)
    print("示例 7: Peer 对象透明性 — IntProxy 算术布线")
    print("-" * 40)

    a = IntProxy(smt_mkvar())
    b = IntProxy(smt_mkvar())
    c = a + 3
    d = c * b
    cond = d > 20

    print(f"  a       = {a}")
    print(f"  c = a+3 = {c}")
    print(f"  d = c*b = {d}")
    print(f"  d > 20  = {cond}")

    s = Solver()
    s.add(cond.formula)
    if s.check() == sat:
        m = s.model()
        print(f"  Z3 求解 d>20: {m}")
