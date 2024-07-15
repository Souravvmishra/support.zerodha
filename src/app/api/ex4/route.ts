import {
  Message as VercelChatMessage,
  StreamingTextResponse,
  createStreamDataTransformer
} from 'ai';
import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { HttpResponseOutputParser } from 'langchain/output_parsers';
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RunnableSequence } from '@langchain/core/runnables'
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import path from 'path';

export const maxDuration = 60;
export const dynamic = 'force-dynamic'

const filePath = path.join(process.cwd(), 'zerodha_articles.txt');

const formatMessage = (message: VercelChatMessage) => {
  return `${message.role}: ${message.content}`;
};

const TEMPLATE = `Answer the user's questions based only on the following context. If the answer is not in the context, reply politely that you do not have that information available.:
==============================
Context: {context}
==============================
Current conversation:
{chat_history}
user: {question}
assistant:`;

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();
    const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage);
    const currentMessageContent = messages[messages.length - 1].content;

    // Load and process the text document
    const loader = new TextLoader(filePath);
    const rawDocs = await loader.load();

    // Split the text into smaller chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const docs = await textSplitter.splitDocuments(rawDocs);

    // Create vector store and index the documents
    const vectorStore = await MemoryVectorStore.fromDocuments(docs, new OpenAIEmbeddings());

    // Retrieve relevant documents based on the question
    const relevantDocs = await vectorStore.similaritySearch(currentMessageContent, 3);

    const prompt = PromptTemplate.fromTemplate(TEMPLATE);

    const model = new ChatOpenAI({
      apiKey: process.env.OPENAI_API_KEY!,
      model: 'gpt-3.5-turbo',
      temperature: 0,
      streaming: true,
      verbose: true,
    });

    const parser = new HttpResponseOutputParser();

    const chain = RunnableSequence.from([
      {
        question: (input) => input.question,
        chat_history: (input) => input.chat_history,
        context: (input) => input.context,
      },
      prompt,
      model,
      parser,
    ]);

    const stream = await chain.stream({
      chat_history: formattedPreviousMessages.join('\n'),
      question: currentMessageContent,
      context: relevantDocs.map(doc => doc.pageContent).join('\n'),
    });

    return new StreamingTextResponse(
      stream.pipeThrough(createStreamDataTransformer()),
    );
  } catch (e: any) {
    return Response.json({ error: e.message }, { status: e.status ?? 500 });
  }
}
