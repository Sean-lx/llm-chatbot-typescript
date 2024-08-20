import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";

export const llm = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: process.env.OPENAI_API_MODEL,
  temperature: 0,
});

export const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY as string,
});
