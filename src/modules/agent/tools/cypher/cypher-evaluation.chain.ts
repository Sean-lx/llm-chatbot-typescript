import { BaseLanguageModel } from "langchain/base_language";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { JsonOutputParser } from "@langchain/core/output_parsers";

// tag::interface[]
export type CypherEvaluationChainInput = {
  question: string;
  cypher: string;
  schema: string;
  errors: string[] | string | undefined;
};
// end::interface[]

// tag::output[]
export type CypherEvaluationChainOutput = {
  cypher: string;
  errors: string[];
};
// end::output[]

// tag::function[]
export default async function initCypherEvaluationChain(
  llm: BaseLanguageModel
) {
  // Prompt template
  const prompt = PromptTemplate.fromTemplate(`
    You are an expert Neo4j Developer evaluating a Cypher statement written by an AI.

    Check that the cypher statement provided below against the database schema to check that
    the statement will answer the user's question.
    Fix the cypher statement according to the errors infomation provided.

    Respond with a JSON object with "cypher" and "errors" keys.
      * "cypher" - the corrected cypher statement
      * "errors" - A list of uncorrectable errors.  For example, if a label,
          relationship type or property does not exist in the schema.
          Provide a hint to the correct element where possible.

    Schema:
    {schema}

    Question:
    {question}

    Cypher Statement:
    {cypher}

    {errors}
  `);

  return RunnableSequence.from<
    CypherEvaluationChainInput,
    CypherEvaluationChainOutput
  >([
    RunnablePassthrough.assign({
      // Convert errors into an LLM-friendly list
      errors: ({ errors }) => {
        if (
          errors === undefined ||
          (Array.isArray(errors) && errors.length === 0)
        ) {
          return "";
        }

        return `[${Array.isArray(errors) ? errors?.join(",") : errors}]`;
      },
    }),
    prompt,
    llm,
    new JsonOutputParser<CypherEvaluationChainOutput>(),
  ]);
}
// end::function[]
