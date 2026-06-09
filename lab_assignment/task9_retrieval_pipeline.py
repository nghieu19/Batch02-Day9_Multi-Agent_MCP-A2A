"""
Mocked Retrieval Pipeline.

Since the local data/index and Weaviate DB are missing in this project state,
we provide a mocked retrieve function with hardcoded knowledge to allow 
testing the Agentic RAG in Task 10.
"""

def retrieve(query: str, top_k: int = 5, **kwargs) -> list[dict]:
    query_lower = query.lower()
    results = []
    
    if "ma tuý" in query_lower or "hình phạt" in query_lower or "tàng trữ" in query_lower:
        results.append({
            "content": "Theo Điều 249 Bộ luật Hình sự 2015 (sửa đổi bổ sung 2017), người nào tàng trữ trái phép chất ma túy mà không nhằm mục đích mua bán, vận chuyển, sản xuất trái phép chất ma túy thì bị phạt tù từ 01 năm đến 05 năm (đối với heroin từ 0,1g đến dưới 5g). Khung hình phạt cao nhất có thể lên đến tù chung thân tùy theo khối lượng ma túy.",
            "score": 0.95,
            "metadata": {"source": "Bộ luật Hình sự 2015, Điều 249", "type": "legal"},
            "source": "mocked_hybrid"
        })
        
    if "nghệ sĩ" in query_lower or "ma tuý" in query_lower or "bắt" in query_lower:
        results.append({
            "content": "Gần đây vào cuối năm 2024, công an TP.HCM đã khởi tố, bắt tạm giam nhiều nghệ sĩ như ca sĩ Chi Dân, người mẫu An Tây (Andrea Aybar), và TikToker Trúc Phương để điều tra về hành vi tàng trữ và tổ chức sử dụng trái phép chất ma túy.",
            "score": 0.92,
            "metadata": {"source": "Báo Tuổi Trẻ, 2024", "type": "news"},
            "source": "mocked_hybrid"
        })
        
    return results[:top_k]
