PROMPT_SR1 = """
Answer the given question. \
Related information is provided in <context> ... </context>. \
You must first conduct reasoning inside <think> ... </think>. \
After reasoning, if you find the provided information is hard to answer the question, you can transform the provided information into a more suitable format by <format: format_name> Your reformatted information </format: format_name>. \
Every time after you transform the provided information into a more suitable format, you must first conduct reasoning inside <think> ... </think>. \

!!! STRICT FORMAT RULES for <format: format_name>: !!!
    + You MUST replace format_name with the real format name, e.g. graph, table, algorithm, etc. \
    + You MUST replace Your reformatted information with a CONCRETE reformatted information that helps answer the original question below. \
    + NEVER copy or paste model descriptions into <format: format_name>. \
    + NEVER output the placeholder format <format: format_name> Your reformatted information </format: format_name>. Always replace both parts correctly. \

### The Descriptions of each Format:

Chunk: \
A chunk is a self-contained summary of one or multiple documents in natural language.

Knowledge Graph: \
A knowledge graph is a structured representation of facts in the form of entities (things) and relations (connections between things), often expressed as triples: (head, relation, tail).

Table: \
A table is a structured way of organizing data into rows and columns. It's commonly used to present information clearly and compactly.

Catalogue: \
A catalogue is a structured, systematically arranged list of items-each described by a consistent set of metadata-that lets readers discover, browse, and retrieve individual entries quickly.

Algorithm: \
An algorithm is a step-by-step procedure for solving a problem or achieving a specific result.

**You can also develop your own format by yourself.** Different documents might need different formats.

If you are ready to answer the question, you can directly provide your final answer inside <answer> ... </answer>, without additional explanation or illustration. \
For example: <answer> Beijing </answer>. \
    + Important: You must not output the placeholder text "<answer> and </answer>" alone. \
    + You must insert your actual answer between <answer> and </answer>, following the correct format. \

<context>
{context}
</context>

Question: {question}\n
"""