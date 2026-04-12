# 下面用 python 的 sexpdata 库，参照 Symbolic Execution of Python Programs with PyExZ3 这篇论文，实现一个对 scheme 语言做符号执行。


import sexpdata
from z3 import *
from copy import deepcopy
from dataclasses import dataclass

# ====================== 1. 路径状态定义（PyExZ3 核心） ======================
@dataclass
class PathState:
    """路径状态：PyExZ3 中 PathState 的 Scheme 适配版"""
    expr: any               # 当前待执行的S表达式
    env: Env                # 当前执行环境
    pc: list                # 路径条件（约束列表）
    solver: Solver          # 独立的Z3求解器上下文（避免路径间污染）
    depth: int = 0          # 路径深度（用于回溯优先级）

    def is_feasible(self):
        """检查当前路径是否可行（约束可满足）"""
        self.solver.push()
        self.solver.add(self.pc)
        res = self.solver.check() == sat
        self.solver.pop()
        return res

# ====================== 2. 环境类（复用并优化） ======================
class Env:
    """Scheme符号执行的环境（词法作用域）"""
    def __init__(self, parent=None):
        self.parent = parent
        self.bindings = {}

    def lookup(self, var):
        if var in self.bindings:
            return self.bindings[var]
        elif self.parent is not None:
            return self.parent.lookup(var)
        else:
            raise NameError(f"Undefined variable: {var}")

    def extend(self, var, value):
        self.bindings[var] = value

    def copy(self):
        """深拷贝环境（回溯时避免状态污染）"""
        new_env = Env(parent=self.parent.copy() if self.parent else None)
        new_env.bindings = deepcopy(self.bindings)
        return new_env

# ====================== 3. 带回溯的Scheme符号执行器 ======================
class SchemeSymbolicExecutorWithBacktrack:
    def __init__(self, search_strategy="dfs"):
        # 路径队列：DFS用栈（list.pop()），BFS用队列（list.pop(0)）
        self.path_queue = []
        self.search_strategy = search_strategy  # dfs/bfs
        self.executed_paths = []  # 记录已执行的路径（回溯结果）

    def parse_scheme(self, scheme_code):
        """解析Scheme代码为S表达式"""
        cleaned_code = "\n".join([line.split(";")[0] for line in scheme_code.split("\n")])
        return sexpdata.loads(cleaned_code)

    def eval_sexp(self, path_state):
        """递归求值S表达式，遇到分支则生成新路径状态（核心回溯逻辑）"""
        sexp = path_state.expr
        env = path_state.env
        pc = path_state.pc
        solver = path_state.solver

        # 1. 字面量处理
        if isinstance(sexp, (int, float)):
            return sexp
        elif isinstance(sexp, sexpdata.Symbol):
            sym_str = sexpdata.dumps(sexp)
            if sym_str == "#t":
                return BoolVal(True)
            elif sym_str == "#f":
                return BoolVal(False)
            return env.lookup(sym_str)
        
        # 2. 复合表达式（列表）
        if not isinstance(sexp, list):
            raise ValueError(f"Invalid S-expression: {sexp}")
        
        op = sexp[0]
        op_str = sexpdata.dumps(op) if isinstance(op, sexpdata.Symbol) else str(op)

        # 3. 条件表达式：if（分支点，生成新路径状态）
        if op_str == "if":
            if len(sexp) != 4:
                raise SyntaxError("if requires 3 arguments: (if cond then else)")
            cond_sexp, then_sexp, else_sexp = sexp[1], sexp[2], sexp[3]
            
            # 求值条件表达式
            cond_val = self.eval_sexp(PathState(cond_sexp, env, pc, solver))
            if not is_bool(cond_val):
                raise TypeError("if condition must be boolean")

            # ---------------- 分支1：条件为真 ----------------
            true_pc = deepcopy(pc)
            true_pc.append(cond_val)
            # 创建新的路径状态（深拷贝环境和求解器，避免污染）
            true_solver = Solver()
            true_solver.add(true_pc)
            true_state = PathState(
                expr=then_sexp,
                env=env.copy(),
                pc=true_pc,
                solver=true_solver,
                depth=path_state.depth + 1
            )
            # 仅将可行路径加入队列
            if true_state.is_feasible():
                self.path_queue.append(true_state)
                print(f"[生成路径] 深度{true_state.depth} | 约束: {true_pc} | 待执行: {then_sexp}")

            # ---------------- 分支2：条件为假 ----------------
            false_pc = deepcopy(pc)
            false_pc.append(Not(cond_val))
            false_solver = Solver()
            false_solver.add(false_pc)
            false_state = PathState(
                expr=else_sexp,
                env=env.copy(),
                pc=false_pc,
                solver=false_solver,
                depth=path_state.depth + 1
            )
            if false_state.is_feasible():
                self.path_queue.append(false_state)
                print(f"[生成路径] 深度{false_state.depth} | 约束: {false_pc} | 待执行: {else_sexp}")

            return None  # if分支通过路径队列回溯执行，不直接返回

        # 4. 变量定义：define
        elif op_str == "define":
            if len(sexp) != 3:
                raise SyntaxError("define requires 2 arguments: (define var val)")
            var_sexp, val_sexp = sexp[1], sexp[2]
            
            # 函数定义语法糖
            if isinstance(var_sexp, list) and len(var_sexp) > 0:
                func_name = sexpdata.dumps(var_sexp[0])
                args = [sexpdata.dumps(a) for a in var_sexp[1:]]
                body = val_sexp
                lambda_sexp = [sexpdata.Symbol("lambda"), args, body]
                val = self.eval_sexp(PathState(lambda_sexp, env, pc, solver))
                env.extend(func_name, val)
            else:
                var_name = sexpdata.dumps(var_sexp)
                val = self.eval_sexp(PathState(val_sexp, env, pc, solver))
                env.extend(var_name, val)
            return val

        # 5. 匿名函数：lambda
        elif op_str == "lambda":
            if len(sexp) != 3:
                raise SyntaxError("lambda requires 2 arguments: (lambda (args) body)")
            args_sexp, body_sexp = sexp[1], sexp[2]
            args = [sexpdata.dumps(a) for a in args_sexp]
            return {"type": "lambda", "args": args, "body": body_sexp, "env": env}

        # 6. 基本运算：+ - * = > <
        elif op_str in ["+", "-", "*", "=", ">", "<"]:
            args = [self.eval_sexp(PathState(a, env, pc, solver)) for a in sexp[1:]]
            for arg in args:
                if not (isinstance(arg, (int, float)) or is_int(arg)):
                    raise TypeError(f"Arithmetic op {op_str} requires numeric args")
            
            if op_str == "+":
                return sum(args[1:], args[0])
            elif op_str == "-":
                return args[0] - sum(args[1:], 0) if len(args) > 1 else -args[0]
            elif op_str == "*":
                result = 1
                for a in args:
                    result *= a
                return result
            elif op_str == "=":
                return args[0] == args[1]
            elif op_str == ">":
                return args[0] > args[1]
            elif op_str == "<":
                return args[0] < args[1]

        # 7. 函数调用
        else:
            func_val = self.eval_sexp(PathState(op, env, pc, solver))
            args_val = [self.eval_sexp(PathState(a, env, pc, solver)) for a in sexp[1:]]
            
            if isinstance(func_val, dict) and func_val["type"] == "lambda":
                lambda_args = func_val["args"]
                lambda_body = func_val["body"]
                lambda_env = func_val["env"]
                
                if len(args_val) != len(lambda_args):
                    raise TypeError(f"Expected {len(lambda_args)} args, got {len(args_val)}")
                
                new_env = Env(parent=lambda_env)
                for arg_name, arg_val in zip(lambda_args, args_val):
                    new_env.extend(arg_name, arg_val)
                
                # 函数体执行：生成新的路径状态（继续回溯）
                call_state = PathState(
                    expr=lambda_body,
                    env=new_env,
                    pc=deepcopy(pc),
                    solver=deepcopy(solver),
                    depth=path_state.depth + 1
                )
                return self.eval_sexp(call_state)
            else:
                raise TypeError(f"Not a function: {op_str}")

    def run(self, initial_expr, initial_env):
        """启动符号执行，按策略探索所有路径（回溯核心入口）"""
        # 初始化初始路径状态
        initial_solver = Solver()
        initial_state = PathState(
            expr=initial_expr,
            env=initial_env,
            pc=[],
            solver=initial_solver,
            depth=0
        )
        self.path_queue.append(initial_state)

        # 路径探索循环（回溯的核心）
        while self.path_queue:
            # 根据搜索策略取路径：DFS（栈）pop()，BFS（队列）pop(0)
            if self.search_strategy == "dfs":
                current_state = self.path_queue.pop()  # 栈：后进先出（回溯）
            else:
                current_state = self.path_queue.pop(0)  # 队列：先进先出

            print(f"\n[执行路径] 深度{current_state.depth} | 约束: {current_state.pc}")
            try:
                # 执行当前路径
                result = self.eval_sexp(current_state)
                # 记录已执行的路径结果
                self.executed_paths.append({
                    "pc": current_state.pc,
                    "depth": current_state.depth,
                    "result": result,
                    "feasible": current_state.is_feasible()
                })
                print(f"[路径结果] 执行结果: {result} | 可行: {current_state.is_feasible()}")
            except Exception as e:
                print(f"[路径失败] 错误: {e}")

        # 输出所有路径总结
        print("\n=== 所有路径执行总结 ===")
        for i, path in enumerate(self.executed_paths):
            print(f"路径{i+1} | 深度{path['depth']} | 约束: {path['pc']} | 结果: {path['result']} | 可行: {path['feasible']}")

# ====================== 4. 测试回溯功能 ======================
if __name__ == "__main__":
    # 1. 初始化执行器（DFS策略，PyExZ3默认DFS）
    executor = SchemeSymbolicExecutorWithBacktrack(search_strategy="dfs")

    # 2. 初始化全局环境（添加符号变量）
    global_env = Env()
    x = Int("x")
    global_env.extend("x", x)
    y = Int("y")
    global_env.extend("y", y)

    # 3. 测试嵌套if（回溯的典型场景）
    scheme_code = """
    (if (> x 5)
        (if (= y 10)
            (+ x y)
            (- x y)
        )
        (if (< y 0)
            (* x y)
            (/ x 2)  ; 注：简化处理，这里/暂用x/2表示
        )
    )
    """
    print("=== 测试嵌套if的路径回溯 ===")
    initial_expr = executor.parse_scheme(scheme_code)
    # 启动符号执行（自动回溯所有路径）
    executor.run(initial_expr, global_env)

    # 4. 求解某条路径的具体值（示例：x>5且y=10）
    print("\n=== 求解路径约束（x>5 ∧ y=10）===")
    solver = Solver()
    solver.add(x > 5, y == 10)
    if solver.check() == sat:
        model = solver.model()
        print(f"x = {model[x]}, y = {model[y]} | x+y = {model[x]+model[y]}")
