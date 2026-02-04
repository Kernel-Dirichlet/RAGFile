# RAGFile
binary file format for RAG based searches, inspired from Hadoop Sequence Files. 


## RAGFile v0.1.0 

Retrieval-Augmented Generation is a popular framework to supplement
respones from Large Language Models (LLMs) with information and context relevant to a user query or input prompt.

There are many retrieval frameworks, which broadly fall under 
vector similarity seaches, graph-based retrieval and agentic retrieval (i.e. an AI agent is responsible for retrieving information from external sources). 

While each of these approaches have strenghts and weaknesses, virtually all of them require layers of software abstraction under the hood. The overhead associated with database and network calls is fairly small, but if we want to scale RAG systems to 10TB+ document stores, we want to optimize as much as we can, where we can. 

RAGFile is a binary file format designed to address these limitations. The format is inspired from Hadoop Sequence Files, which are optimized for disk I/O in HDFS workloads. 


### RAGFile features

1) Storage-efficiency 
   - we want to cram as much data into the file as possible

2) Retrieval method agnostic
   - RAG strategies evolve by the day and the binary file should not be explicitly tied to any one strategy 

3) Fast retrieval w/different strategies
   - in order to retrieve data efficiently, we will need to support
     indexing (initial organization & layout) that will enable this. 

   - Effective indexing can also be achieved by leveraging *hardware* in addition to clever software tricks. 



### RAGFile Schema

   Below is the schema for RAGFiles as of v0.1.0, in order.
   1 - 3 are mandatory, 4 onward are semi-optional in that no section is strictly required, but the existence of one might mandate the existence of another. When this occurs, they will be denoted by sub-sections. 

   1) **"RAGFILE" header** (verification of file)
   2) **RAGFile version** (major,minor,patch)
   3) **Endianess** (1 - Big Endian, 0 - Little Endian)
   4) **"Index strategy sections"**
      
      This section of the RAGFile contains any number of 

      {strategy}-({start_byte,end_byte}) pairs. 

      For each strategy, the start-byte denotes the start of that indexing strategy's section and the end-byte denotes the end of that section. 

      There is always a single \x00 between each 
      {strategy}-({start_byte,end_byte}) pairs. 

      example: 

      keyword-(24,4096) vector-(4098,8192) (note \x00 is at byte position 4097)

      below are supported formats in v0.1.0 
   
   5) **Keyword-content pairs**
      
       I)  start-byte (the first byte that starts the first keyword-content pair)

       II) end-byte (the last byte which is the last keyword-content pair)

       III) padding - an integer which represents how many null \x00 bytes to pad in between each keyword-content pair and is designed to be read in chunks. For now, padding must be 4,8, or 16.  

         keyword-content pairs are formatted as {keyword}-{content} (with a single "-" in between). 

        When we search for keywords, we have the start and end byte boundaries. This has the benefit of supporting storage of other indexed data (ex. vector and graph) w/o impacting the retrieval speed of keyword-   content pairs.

      Also do note that content can be variable length, and there are approaches to narrow the search space further which will be a future feature. 
    

   6)  **Embedding-content pairs**
    
    I) precision - an integer representing the numerical precision of the embedding vectors. This is used to 
        
    II) start-byte (the first byte that starts the first keyword-content pair)

    III) end-byte (the last byte which is the last keyword-content pair)

    IV) padding - an integer which represents how many null \x00 bytes to pad in between each keyword-content pair and is designed to be read in chunks. For now, padding must be 4,8, or 16.  

   embedding-content pairs are formed with the same structure as keyword-content pairs w/ {embedding}-{content}

   Future updates will include graph and hypergraph formats which will encode node, edge, and hyperedge features where relevant. 

        

      
  





