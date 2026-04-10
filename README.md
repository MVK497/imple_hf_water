# Simple Hartree-Fock, UHF, And MP2 Teaching Example

这个项目现在扩展成了一个更适合教学和继续修改的小型电子结构小程序：

- 支持任意分子坐标输入
- 支持 `STO-3G` 和 `6-31G(d)`
- 保留“自己写 SCF 主循环、电子积分交给 `PySCF`”的结构
- 默认使用 `DIIS` 加速 SCF 收敛
- 新增支持开壳层体系的 `UHF`
- 新增基于收敛 RHF 轨道的 `RMP2`
- 按模块拆分，便于阅读和改写

## 安装依赖

```bash
cd /Users/roxy/ROXY_Projects/projects/simple_hf_water
python3 -m pip install -r requirements.txt
```

## 目录结构

- `rhf_sto3g_water.py`: 顶层入口脚本
- `simple_hf/geometry.py`: 分子坐标输入与默认示例
- `simple_hf/rhf.py`: 分子构建与 RHF 主循环
- `simple_hf/uhf.py`: 最小 `UHF` 实现
- `simple_hf/mp2.py`: 最小 `RMP2` 实现
- `simple_hf/cli.py`: 命令行参数与结果打印
- `examples/water.xyz`: 水分子示例输入

## 运行示例

默认计算水分子，基组默认是 `sto-3g`：

```bash
python3 rhf_sto3g_water.py
```

计算水分子的 `MP2` 能量：

```bash
python3 rhf_sto3g_water.py --method mp2
```

计算开壳层体系的 `UHF`，例如氢原子：

```bash
python3 rhf_sto3g_water.py --method uhf --geometry "H 0 0 0" --spin 1
```

从 XYZ 文件读取分子：

```bash
python3 rhf_sto3g_water.py --xyz examples/water.xyz
```

直接在命令行输入几何：

```bash
python3 rhf_sto3g_water.py --geometry "O 0 0 0; H 0 -0.757160 0.586260; H 0 0.757160 0.586260"
```

改用 `6-31G(d)`：

```bash
python3 rhf_sto3g_water.py --xyz examples/water.xyz --basis '6-31G(d)'
```

计算 `6-31G(d)` 下的 `MP2`：

```bash
python3 rhf_sto3g_water.py --xyz examples/water.xyz --basis '6-31G(d)' --method mp2
```

也可以用 `6-31G*` 作为等价别名：

```bash
python3 rhf_sto3g_water.py --xyz examples/water.xyz --basis '6-31G*'
```

查看 SCF 每一步能量：

```bash
python3 rhf_sto3g_water.py --xyz examples/water.xyz --show-history
```

关闭 DIIS，观察朴素 SCF 与 DIIS 的区别：

```bash
python3 rhf_sto3g_water.py --xyz examples/water.xyz --basis '6-31G(d)' --no-diis
```

## 说明

- 这个程序实现的是封闭壳层 `RHF`
- `UHF` 支持 `spin != 0` 的开壳层体系
- `MP2` 建立在收敛的封闭壳层 `RHF` 结果之上
- `RHF` 和 `MP2` 当前仍要求 `spin = 0`
- 如果总电子数是奇数，程序会报错提醒
- XYZ 文件输入默认按 `Angstrom` 处理
- 当前只接受 `sto-3g`、`6-31G(d)` 和 `6-31G*` 这几种写法
- 默认最大 SCF 迭代步数是 `100`
- 默认开启 `DIIS`，`--diis-space` 默认是 `6`
- `DIIS` 的误差矩阵使用 `FDS - SDF`
- `UHF` 中分别对 `alpha` 和 `beta` Fock 矩阵构造误差，并用联合 DIIS 外推
- `MP2` 部分用 `numpy.einsum` 做 `AO -> MO` 双电子积分变换，适合小体系教学演示
