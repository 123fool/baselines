# 数据盘点说明（2026-04-14）

## 目的

统一说明服务器端 AD/MCI/CN 的数据规模，澄清“96 条 AD 记录”仅对应纵向子集，不代表 AD 全量。

## 统计口径

- 服务器根目录：`/home/wangchong/data/fwz/data`
- 目录级别定义：
  - 一级被试目录（每个被试一个文件夹）
  - 二级时间点目录（每个时间点一个文件夹）
- 计数定义：
  - `subjects`：被试数（一级目录数）
  - `records`：记录数（二级时间点目录总数）
- 分组目录：
  - `*_non_longitudinal`：单时间点被试
  - `*_longitudinal`：多时间点被试

## 统计结果

### AD

- non_longitudinal：subjects=90，records=90
- longitudinal：subjects=41，records=96
- overlap（两目录被试交集）=0
- 去重后合计：subjects=131，records=186

### MCI

- non_longitudinal：subjects=247，records=247
- longitudinal：subjects=190，records=661
- overlap（两目录被试交集）=0
- 去重后合计：subjects=437，records=908

### CN

- non_longitudinal：subjects=536，records=536
- longitudinal：subjects=50，records=165
- overlap（两目录被试交集）=0
- 去重后合计：subjects=586，records=701

## 结论

1. AD 的 96 条记录仅是 `ad_longitudinal` 的记录数。
2. AD 全量记录应按 `ad_non_longitudinal + ad_longitudinal` 统计，即 186 条。
3. 纵向与非纵向被试集合互斥（交集为 0），可直接相加得到去重后总数。

## 与训练 CSV 的关系

- `/home/wangchong/data/fwz/data/diagnosis_categorized/ad_brlp_innovation.csv` 行数为 96，来源对应纵向 AD 子集（可形成多时间点训练样本）。
- 因此该 CSV 不能代表 AD 全量，只代表当前纵向训练链路使用的数据子集。
