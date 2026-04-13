# Simple Electronic Structure Teaching Example

这个项目是一个面向学习的量子化学小程序，当前已经支持：

- `RHF`
- `UHF`
- `RKS`
- `UKS`
- `RMP2`
- `UMP2`
- `RHF-based CCSD`
- `RHF/UHF/RKS/UKS` 几何优化
- 键长刚性扫描 / 柔性扫描
- 键角刚性扫描 / 柔性扫描
- 二面角刚性扫描 / 柔性扫描

整体思路是：

- `SCF` 主循环尽量自己写，便于学习
- 电子积分、解析梯度、相关方法求解器等复杂底层能力适当调用 `PySCF`
- 所有功能都尽量通过统一的命令行入口使用

## 安装

```bash
cd /Users/roxy/ROXY_Projects/projects/simple_hf_water
python3 -m pip install -r requirements.txt
```

## 项目结构

- `rhf_sto3g_water.py`: 顶层入口脚本
- `simple_hf/geometry.py`: 几何输入、坐标变换、键长/键角/二面角工具
- `simple_hf/rhf.py`: `RHF` 与 `DIIS`
- `simple_hf/uhf.py`: `UHF` 与 `<S^2>`
- `simple_hf/rks.py`: `RKS`
- `simple_hf/uks.py`: `UKS` 与 `<S^2>`
- `simple_hf/mp2.py`: `RMP2`
- `simple_hf/ump2.py`: `UMP2`
- `simple_hf/ccsd.py`: `CCSD`
- `simple_hf/optimize.py`: `RHF/UHF/RKS/UKS` 几何优化
- `simple_hf/scan.py`: 内坐标扫描模块
- `examples/water.xyz`: 水分子示例
- `examples/oh_radical.xyz`: `OH` 自由基示例
- `examples/h2o2.xyz`: 过氧化氢示例，可用于二面角扫描

## 支持的基组

当前为了保持教学范围清晰，只接受：

- `sto-3g`
- `6-31G(d)`
- `6-31G*`

其中 `6-31G*` 会自动映射到 `6-31G(d)`。

## 几何输入

默认不提供输入时，程序使用水分子示例。

也可以从 `XYZ` 文件读取：

```bash
python3 rhf_sto3g_water.py --xyz examples/water.xyz
```

或者直接在命令行写几何：

```bash
python3 rhf_sto3g_water.py --geometry "O 0 0 0; H 0 -0.757160 0.586260; H 0 0.757160 0.586260"
```

## 单点能计算

`RHF`：

```bash
python3 rhf_sto3g_water.py --method rhf
```

`UHF`：

```bash
python3 rhf_sto3g_water.py --method uhf --geometry "H 0 0 0" --spin 1
```

`RKS`：

```bash
python3 rhf_sto3g_water.py --method rks --xc b3lyp
```

`UKS`：

```bash
python3 rhf_sto3g_water.py --method uks --xc pbe --xyz examples/oh_radical.xyz --spin 1
```

`RMP2`：

```bash
python3 rhf_sto3g_water.py --method mp2
```

`UMP2`：

```bash
python3 rhf_sto3g_water.py --method ump2 --xyz examples/oh_radical.xyz --spin 1
```

`CCSD`：

```bash
python3 rhf_sto3g_water.py --method ccsd
```

改用 `6-31G(d)`：

```bash
python3 rhf_sto3g_water.py --xyz examples/water.xyz --basis '6-31G(d)' --method ccsd
```

常见的 `DFT` 泛函可以通过 `--xc` 指定，例如：

- `lda,vwn`
- `pbe`
- `b3lyp`

例如：

```bash
python3 rhf_sto3g_water.py --method rks --xc pbe --basis '6-31G(d)'
```

## 几何优化

当前几何优化支持：

- `RHF`
- `UHF`
- `RKS`
- `UKS`

水分子 `RHF` 优化：

```bash
python3 rhf_sto3g_water.py --method rhf --optimize
```

`OH` 自由基 `UHF` 优化：

```bash
python3 rhf_sto3g_water.py --method uhf --xyz examples/oh_radical.xyz --spin 1 --optimize
```

水分子 `RKS/B3LYP` 优化：

```bash
python3 rhf_sto3g_water.py --method rks --xc b3lyp --optimize
```

`OH` 自由基 `UKS/PBE` 优化：

```bash
python3 rhf_sto3g_water.py --method uks --xc pbe --xyz examples/oh_radical.xyz --spin 1 --optimize
```

查看优化历史：

```bash
python3 rhf_sto3g_water.py --method rhf --optimize --show-history
```

## 内坐标扫描

当前扫描功能支持三类内坐标：

- `bond`
- `angle`
- `dihedral`

如果不显式写 `--scan-coordinate`，默认使用 `angle`。

并支持两种模式：

- `rigid`: 只改目标内坐标，其他坐标固定
- `relaxed`: 对每个扫描点做受约束优化，让其余自由度松弛

原子编号使用 **1-based** 写法。

### 键长扫描

键长使用两个原子，例如 `1,2`。

刚性键长扫描：

```bash
python3 rhf_sto3g_water.py \
  --method rhf \
  --scan rigid \
  --scan-coordinate bond \
  --scan-atoms 1,2 \
  --scan-start 0.85 \
  --scan-stop 1.15 \
  --scan-points 7
```

柔性键长扫描：

```bash
python3 rhf_sto3g_water.py \
  --method rhf \
  --scan relaxed \
  --scan-coordinate bond \
  --scan-atoms 1,2 \
  --scan-start 0.85 \
  --scan-stop 1.15 \
  --scan-points 7
```

对于键长扫描，`--scan-start` 和 `--scan-stop` 使用几何输入的坐标单位：

- 默认是 `Angstrom`
- 如果你用 `--unit Bohr` 输入几何，那么这里也按 `Bohr`

### 键角扫描

键角使用三个原子 `i-j-k`。

例如水分子的 `H-O-H` 角，如果原子顺序是 `O H H`，那么角就是 `2,1,3`。

刚性键角扫描：

```bash
python3 rhf_sto3g_water.py \
  --method rhf \
  --scan rigid \
  --scan-coordinate angle \
  --scan-atoms 2,1,3 \
  --scan-start 95 \
  --scan-stop 115 \
  --scan-points 5
```

柔性键角扫描：

```bash
python3 rhf_sto3g_water.py \
  --method rhf \
  --scan relaxed \
  --scan-coordinate angle \
  --scan-atoms 2,1,3 \
  --scan-start 95 \
  --scan-stop 115 \
  --scan-points 5
```

### 二面角扫描

二面角使用四个原子 `i-j-k-l`。

刚性二面角扫描：

```bash
python3 rhf_sto3g_water.py \
  --method rhf \
  --scan rigid \
  --scan-coordinate dihedral \
  --xyz examples/h2o2.xyz \
  --scan-atoms 1,2,3,4 \
  --scan-start -180 \
  --scan-stop 180 \
  --scan-points 13
```

柔性二面角扫描：

```bash
python3 rhf_sto3g_water.py \
  --method rhf \
  --scan relaxed \
  --scan-coordinate dihedral \
  --xyz examples/h2o2.xyz \
  --scan-atoms 1,2,3,4 \
  --scan-start -180 \
  --scan-stop 180 \
  --scan-points 13
```

### 柔性扫描支持的方法

柔性扫描当前支持：

- `RHF`
- `UHF`
- `RKS`
- `UKS`

因为它内部需要解析梯度和几何优化。

`DFT` 扫描时也可以直接指定泛函，例如：

```bash
python3 rhf_sto3g_water.py \
  --method rks \
  --xc b3lyp \
  --scan rigid \
  --scan-coordinate angle \
  --scan-atoms 2,1,3 \
  --scan-start 95 \
  --scan-stop 115 \
  --scan-points 5
```

### 导出扫描结果

可以把扫描结果导出成 `CSV`：

```bash
python3 rhf_sto3g_water.py \
  --method rhf \
  --scan rigid \
  --scan-coordinate angle \
  --scan-atoms 2,1,3 \
  --scan-start 95 \
  --scan-stop 115 \
  --scan-points 5 \
  --scan-output water_scan.csv
```

输出列包括：

- `index`
- `coordinate_type`
- `target_value`
- `actual_value`
- `value_unit`
- `energy_hartree`
- `converged`

### 柔性扫描中的约束强度

柔性扫描使用一个简化的谐振子惩罚项来约束目标角度：

```bash
python3 rhf_sto3g_water.py \
  --method rhf \
  --scan relaxed \
  --scan-coordinate angle \
  --scan-atoms 2,1,3 \
  --scan-start 95 \
  --scan-stop 115 \
  --scan-points 5 \
  --constraint-k 50
```

`--constraint-k` 的单位取决于扫描类型：

- 对 `bond` 是 `Eh/Bohr^2`
- 对 `angle` 和 `dihedral` 是 `Eh/rad^2`

## 常用参数

- `--charge`: 分子总电荷
- `--spin`: `2S = N(alpha) - N(beta)`
- `--show-history`: 打印 `SCF` 或几何优化历史
- `--no-diis`: 关闭 `DIIS`
- `--diis-space`: 设置 `DIIS` 子空间大小
- `--scan-coordinate`: 选择 `bond / angle / dihedral`
- `--scan-atoms`: 指定扫描涉及的原子编号
- `--scan-output`: 导出扫描结果 CSV
- `--max-iter`: 最大 `SCF` 迭代数
- `--opt-max-steps`: 最大几何优化步数
- `--grad-tol`: 几何优化梯度阈值
- `--max-step-size`: 几何优化最大步长，单位 `Bohr`

## 方法说明

- `RHF`、`RMP2`、`CCSD` 当前要求 `spin = 0`
- `UHF` 和 `UMP2` 支持开壳层体系
- `UHF` 会输出 `<S^2>`、理论 `<S^2>` 和自旋污染
- `RMP2` 使用 `AO -> MO` 双电子积分变换
- `UMP2` 会把相关能拆成 `aa`、`ab`、`bb` 三部分
- `CCSD` 当前复用我们自己的 `RHF` 参考轨道，再调用 `PySCF` 的 `CCSD` 求解器
- 几何优化使用 `PySCF` 解析梯度和一个简化的 `BFGS + backtracking` 循环
- 刚性扫描是教学化版本：
  - 键长扫描只移动第二个原子
  - 键角扫描只移动第三个原子
  - 二面角扫描只旋转第四个原子
- 柔性扫描本质上是“受约束优化”的教学版本：
  - 用谐振子惩罚项近似约束
  - 不是严格的拉格朗日乘子实现

## 适合拿来练习的下一步

- 扩展为键长扫描
- 扩展为二面角扫描
- 给优化加入频率分析
- 给扫描结果自动生成能量曲线图
- 给 `CCSD` 再加 `CCSD(T)`
