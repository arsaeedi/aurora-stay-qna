

from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with the retrieved context
# (same rules as in the notebook, so the deployed app behaves like the evaluated one)
def generate(query, retrieved_chunks, model) -> str:

    # Rules that keep the answer grounded in AuroraStay's policy documents
    rules = """
    1. Answer only from the retrieved AuroraStay policy information. Do not use general knowledge about how hotels usually work.
    2. Quote figures exactly as written: fees, deadlines, times, limits and property codes (for example ASH-CHI Loop).
    3. If the answer depends on the property, the rate plan (Flexible, Advance Purchase, member/promotional, group) or the booking channel, say so and give the rule for each case the information covers.
    4. Include the conditions, exceptions and approvals that apply (for example manager approval, availability, documentation needed).
    5. If the property or detail asked about is not in the retrieved information, say it is not covered and advise staff to check the live property profile or escalate to the manager on duty. Never guess.
    6. If two pieces of information conflict, point out the conflict and recommend escalation.
    7. Start with a direct one- or two-sentence answer, then list the supporting conditions as short bullet points.
    8. Leave out details the question did not ask about.
    """

    prompt = f"""
    You are an assistant that helps AuroraStay Hotels staff answer guest questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    {rules}

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
