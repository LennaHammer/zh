# PeerCheck 极简版
require 'z3'

# ========== SMT 接口 ==========
module SMT
  def self.var = Z3.Int("s#{rand(1000)}")
  def self.int(n) = n # Z3.int(n)
  def self.bool(b) = b # Z3.bool(b)
end

# ========== 统一代理基类：反射自动转发所有运算符 ==========
class SymbolicProxy
  attr_reader :expr

  def initialize(expr)
    @expr = expr
  end

  # 反射：自动捕获 + - * / == < > <= >= 等所有运算符
  def method_missing(name, *args)
    p method_missing:name
    return super unless %i[+ - * / == < > <= >=].include?(name)

    other = args.first
    rhs = other.is_a?(SymbolicProxy) ? other.expr : SMT.int(other)
    res = @expr.send(name, rhs)
    p res.class
    res.is_a?(Z3::BoolExpr) ? BoolProxy.new(res) : IntProxy.new(res)
  end
  def ==(other)
    return BoolProxy.new(Z3.Eq(@expr, SMT.int(other))).to_bool
  end

  # 反射：声明支持的运算符
  def respond_to_missing?(name, *)
    %i[+ - * / == < > <= >=].include?(name)
  end

  # 支持 1 + proxy 这种反向运算
  def coerce(other)
    [IntProxy.new(SMT.int(other)), self]
  end
end

# ========== 整数代理 ==========
class IntProxy < SymbolicProxy
end

$path = []
$cond = []
# ========== 布尔代理 + 路径控制 ==========
class BoolProxy < SymbolicProxy
 

  # 反射：Ruby 条件判断自动调用 !
  def !
    BoolProxy.new(Z3.not(@expr))
  end

  # 核心：控制条件分支路径
  # 反射：条件分支自动触发
  def to_bool
    p :to_bool
  #   t = Z3.sat? Z3.and(*$cond, @expr)
  #   f = Z3.sat? Z3.and(*$cond, Z3.not(@expr))
	# # 强制分支（只有一条可行）
  #   return true  if t && !f
  #   return false if f && !t
	# 深度限制
    raise 'depth' if $path.size >= 10
	# 按预设路径执行
    if $path.size > $cond.size
      b = $path[$cond.size]
      $cond << (b ? @expr : !@expr)
      return b
    end
	# 默认走 true 分支
    $path << true
    $cond << @expr
    true
  end
end

# ========== 符号执行引擎：路径循环 ==========
module PeerCheck
  def self.sym = IntProxy.new(SMT.var)

  def self.run(fn, *args)
    $path.clear
    loop do
      $cond.clear
      begin
      fn.call(*args) 
       rescue
        puts cond:$cond,path:$path
        solver = Z3::Solver.new
        solver.assert Z3.And(*$cond)
        if solver.satisfiable? 
          p solver.model
        end
        # raise # 
        puts "⚠️ 异常: #{$!}"
      end
      # 回溯分支：移除已探索的 false 分支
      while !$path.empty? && !$path.last
        $path.pop
      end
      break if $path.empty?
      # 翻转最后一个分支
      $path[-1] = false
    end
  end
end

# ========== 测试用例 ==========
def succ(x)
  raise "bug!" if x == 42768
  x + 1
end

def abs(x)
  x >= 0 ? x : -x
end
 
def quick_sort(arr)
  return arr if arr.size <= 1
  pivot = arr[0]
  less  = arr[1..].select { |v| v <= pivot }
  more  = arr[1..].select { |v| v > pivot }
  quick_sort(less) + [pivot] + quick_sort(more)
end

# 运行
PeerCheck.run(method(:succ), PeerCheck.sym)
PeerCheck.run(method(:abs), PeerCheck.sym)
p :ok