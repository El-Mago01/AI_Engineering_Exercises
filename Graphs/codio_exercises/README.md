Interesting learnings from this exercise
- Use TypedDict to describe a dictionary's expected keys and value types.
- Pass this definition to StateGraph as the schema for the shared state that graph nodes read and update.
- TypedDict describes the structure; the actual state is an ordinary dictionary.
- Working with StateGraph: adding nodes, adding edges from START to END
- Compiling a graph using compile() that checks and prepares the graph
- invoke() to run the graph from START until END
- working with conditional edge nodes that creates a conditional edge, almost like an if, then for graph flows
- 