import { FunctionDeclaration as GoogleTool } from "@google/genai"
import { ModelInfo, VertexModelId, vertexDefaultModelId, vertexModels } from "@shared/api"
import { buildExternalBasicHeaders } from "@/services/EnvUtils"
import { ClineStorageMessage } from "@/shared/messages/content"
import { ClineTool } from "@/shared/tools"
import { ApiHandler, CommonApiHandlerOptions } from "../"
import { withRetry } from "../retry"
import { ApiStream } from "../transform/stream"
import { GeminiHandler } from "./gemini"
import { GoogleAuth } from "google-auth-library"

// We use basic HTTP requests for the discoveryengine API since grpc is failing to compile on android
import axios from "axios"

interface VertexHandlerOptions extends CommonApiHandlerOptions {
	vertexProjectId?: string
	vertexRegion?: string
	apiModelId?: string
	thinkingBudgetTokens?: number
	geminiApiKey?: string
	geminiBaseUrl?: string
	ulid?: string
	reasoningEffort?: string
}

export class VertexHandler implements ApiHandler {
	private geminiHandler: GeminiHandler | undefined
	private options: VertexHandlerOptions

	constructor(options: VertexHandlerOptions) {
		this.options = options
	}

	private ensureGeminiHandler(): GeminiHandler {
		if (!this.geminiHandler) {
			try {
				this.geminiHandler = new GeminiHandler({
					...this.options,
					isVertex: true,
				})
			} catch (error: any) {
				throw new Error(`Error creating Vertex AI Gemini handler: ${error.message}`)
			}
		}
		return this.geminiHandler
	}

	@withRetry()
	async *createMessage(systemPrompt: string, messages: ClineStorageMessage[], tools?: ClineTool[]): ApiStream {
		// Convert to basic text question
		const userQuery = messages[messages.length - 1].content as any
		const queryText = typeof userQuery === "string" ? userQuery : 
			Array.isArray(userQuery) ? (userQuery.find(u => u.type === "text")?.text || "Explain this code") :
			"Explain this code"
			
		const dataStoreId = process.env.VERTEX_AI_DATA_STORE_ID
		if (!dataStoreId) {
			// Fallback to regular gemini vertex search if they don't have GenAI App Builder configured
			console.log("[*] Note: VERTEX_AI_DATA_STORE_ID not set. Using standard Vertex API instead of GenAI App Builder credits.")
			const geminiHandler = this.ensureGeminiHandler()
			yield* geminiHandler.createMessage(systemPrompt, messages, tools as GoogleTool[])
			return
		}

		console.log(`[*] Initializing Vertex AI Grounded Search (Data Store: ${dataStoreId})`)
		console.log("[*] Note: This retrieval process consumes your $1,000 GenAI App Builder credits.")

		try {
			const auth = new GoogleAuth({
				scopes: ['https://www.googleapis.com/auth/cloud-platform']
			});
			const client = await auth.getClient();
			const projectId = await auth.getProjectId();
			const token = await client.getAccessToken() as any;
			const location = process.env.VERTEX_AI_LOCATION || "global"

			// We need to construct a ConversationalRetrieval query
			const url = `https://discoveryengine.googleapis.com/v1alpha/projects/${projectId}/locations/${location}/collections/default_collection/engines/${dataStoreId}/conversations/-:converse`

			// Format the payload 
			const payload = {
				query: {
					text: queryText
				},
				// We can configure grounding, safe responses, etc
				summarySpec: {
					summaryResultCount: 3,
					ignoreAdversarialQuery: true,
					includeCitations: true
				}
			}

			// Just yield an initial start
			yield {
				type: "text",
				text: `Thinking... (Querying GenAI App Builder: Data Store ${dataStoreId})\n\n`
			}

			// Make the REST request
			const response = await axios.post(url, payload, {
				headers: {
					'Authorization': `Bearer ${token.token}`,
					'Content-Type': 'application/json'
				}
			})

			const data = response.data
			const answer = data.reply?.summary?.summaryText || data.reply?.reply || "I couldn't find an answer in the data store."

			yield {
				type: "text",
				text: answer
			}
			
			// Extract citations
			if (data.reply?.summary?.summaryWithMetadata?.references) {
				const refs = data.reply.summary.summaryWithMetadata.references
				if (refs.length > 0) {
					yield {
						type: "text",
						text: "\n\nSources:\n"
					}
					for (const ref of refs) {
						if (ref.title || ref.uri) {
							yield {
								type: "text",
								text: `- ${ref.title || ref.uri}\n`
							}
						}
					}
				}
			}
			
		} catch (error: any) {
			console.error("Error connecting to Vertex AI Discovery Engine:", error?.response?.data || error)
			
			yield {
				type: "text",
				text: `Error connecting to GenAI App Builder: ${error?.message || "Unknown error"}\nMake sure your VERTEX_AI_DATA_STORE_ID is valid and you've run 'gcloud auth application-default login'`
			}
			
			// Fallback
			yield {
				type: "text",
				text: "\n\nFalling back to standard Vertex AI..."
			}
			
			const geminiHandler = this.ensureGeminiHandler()
			yield* geminiHandler.createMessage(systemPrompt, messages, tools as GoogleTool[])
		}
	}

	getModel(): { id: VertexModelId; info: ModelInfo } {
		const modelId = this.options.apiModelId
		if (modelId && modelId in vertexModels) {
			const id = modelId as VertexModelId
			return { id, info: vertexModels[id] }
		}
		return {
			id: vertexDefaultModelId,
			info: vertexModels[vertexDefaultModelId],
		}
	}
}
