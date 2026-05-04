import scanpy as sc

# 读取 .h5ad 文件
adata = sc.read_h5ad("D:/GSE/GSE169379_MIBC_snSeq.h5ad")




# 提取用于构建先验分布的列（细胞类型 + 患者编号）
adata.obs[['celltype', 'Patient']].to_csv("D:/GSE/snSeq.csv")
