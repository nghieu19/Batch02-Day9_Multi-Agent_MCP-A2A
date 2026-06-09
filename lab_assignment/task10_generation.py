"""
Task 10 — Generation Có Citation (Agentic RAG).

Nâng cấp chatbot bằng LangGraph `create_react_agent`, cho phép LLM tự động 
quyết định khi nào cần tra cứu tài liệu thông qua tool `search_vietnam_law`.
"""

import asyncio
import os
import sys
import warnings

# Suppress LangGraph deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")

# Thêm root directory vào sys.path để import từ common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from common.llm import get_llm
from lab_assignment.task9_retrieval_pipeline import retrieve

load_dotenv()


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle".
    """
    if len(chunks) <= 2:
        return chunks
    front = [chunks[i] for i in range(0, len(chunks), 2)]
    back = [chunks[i] for i in range(1, len(chunks), 2)]
    back.reverse()
    return front + back


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string có source labels cho citation.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", f"Source {i}")
        doc_type = meta.get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


@tool
def search_vietnam_law(query: str, top_k: int = 5) -> str:
    """Tra cứu các văn bản pháp luật, tin tức và tài liệu liên quan đến luật pháp Việt Nam.
    
    Args:
        query: Câu hỏi hoặc từ khóa tìm kiếm (VD: 'tội tàng trữ ma túy', 'mức phạt đánh bạc')
        top_k: Số lượng tài liệu trả về
    """
    print(f"\n  [Tool Execute] Đang tra cứu cơ sở dữ liệu với query: '{query}'...")
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return "Không tìm thấy thông tin nào liên quan trong cơ sở dữ liệu pháp luật."
    
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    return context


SYSTEM_PROMPT = """Bạn là một trợ lý pháp lý AI chuyên nghiệp về pháp luật Việt Nam.
Nhiệm vụ của bạn là giải đáp các câu hỏi pháp lý một cách chi tiết và chính xác.

Quy tắc quan trọng (BẮT BUỘC):
1. Bạn CẦN gọi tool `search_vietnam_law` để tra cứu thông tin nếu câu hỏi liên quan đến pháp luật hoặc sự kiện thực tế. Bạn không được tự suy đoán hoặc trả lời dựa trên kiến thức có sẵn nếu chưa tra cứu.
2. Với mỗi thông tin, con số, điều luật, hay sự kiện bạn cung cấp, BẮT BUỘC phải kèm theo trích dẫn nguồn (citation) trong ngoặc vuông dựa trên kết quả trả về từ tool. Ví dụ: [Luật Phòng chống ma tuý 2021, Điều 3] hoặc [VnExpress, 2024]. Trích dẫn nguồn (Source) được hiển thị rõ trong context.
3. Nếu kết quả tìm kiếm không có thông tin để trả lời, hãy nói rõ 'Tôi không tìm thấy thông tin này trong cơ sở dữ liệu hiện có' thay vì tự bịa ra thông tin.
4. Trả lời bằng tiếng Việt, cấu trúc thành các đoạn văn rõ ràng.
"""

async def generate_with_citation(query: str) -> str:
    """
    End-to-end RAG generation với citation sử dụng ReAct Agent.
    """
    llm = get_llm()
    tools = [search_vietnam_law]
    agent = create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)
    
    inputs = {"messages": [("user", query)]}
    
    final_answer = ""
    async for chunk in agent.astream(inputs, stream_mode="updates"):
        for node_name, update in chunk.items():
            messages = update.get("messages", [])
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  [Agent] Quyết định gọi tool: {tc['name']} với args: {tc['args']}")
                
                elif msg.type == "ai" and msg.content:
                    final_answer = msg.content
    return final_answer


async def main():
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
    ]
    
    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        answer = await generate_with_citation(q)
        print(f"\nA: {answer}")


if __name__ == "__main__":
    asyncio.run(main())