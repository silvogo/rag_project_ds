#from src.chain import load_chain

if __name__ == '__main__':
    chain = load_chain()
    query = "Can you please tell me the score of the customer A?"
    response = chain.invoke({"input": query})
    print(response['answer'])