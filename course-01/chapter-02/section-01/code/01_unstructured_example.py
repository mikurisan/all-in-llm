from unstructured.partition.auto import partition
from collections import Counter


# PDF file path
pdf_path = "./data/rag.pdf"

# 使用 Unstructured 加载并解析 PDF file
elements = partition(
    filename=pdf_path,
    content_type="application/pdf"
)

print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

# 统计 elements type
types = Counter(e.category for e in elements)
print(f"元素类型: {dict(types)}")

# 显示 all elements
print("\n所有元素:")
for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)
    print("=" * 60)