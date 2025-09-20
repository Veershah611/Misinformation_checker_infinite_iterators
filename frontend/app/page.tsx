"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ThemeToggle } from "@/components/theme-toggle"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import {
  Loader2,
  Upload,
  CheckCircle,
  XCircle,
  AlertTriangle,
  HelpCircle,
  Search,
  Link as LinkIcon,
  FileText,
  ThumbsUp,
  ThumbsDown,
  ShieldAlert,
  BookUser,
  Link2,
  Send,
} from "lucide-react"

// Defines the structure for the fact-check result from the API
interface FactCheckResult {
  claim_analysis: {
    verdict: string
    score: number
    explanation: string
  }
  categorized_points?: {
    points_supporting_truthfulness: string[]
    points_refuting_the_claim: string[]
  }
  risk_assessment?: {
    possible_consequences: string[]
  }
  public_guidance_and_resources?: {
    tips_to_identify_similar_scams: string[]
    official_government_resources: {
      relevant_agency_website: string
      national_helpline_number: string
    }
  }
  evidence_log?: {
    external_sources: Array<{
      source_name: string
      url: string
    }>
  }
  error?: string
}

export default function AIFactChecker() {
  const [claim, setClaim] = useState("")
  const [url, setUrl] = useState("")
  const [files, setFiles] = useState<FileList | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [results, setResults] = useState<FactCheckResult | null>(null)
  const [activeTab, setActiveTab] = useState("text")
  const [replies, setReplies] = useState<string[]>([])
  const [isReplyLoading, setIsReplyLoading] = useState(false)
  
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL

  const getVerdictIcon = (verdict: string) => {
    const normalizedVerdict = verdict?.toLowerCase() ?? ""
    if (normalizedVerdict.includes("true")) return <CheckCircle className="w-5 h-5" />
    if (normalizedVerdict.includes("false")) return <XCircle className="w-5 h-5" />
    if (normalizedVerdict.includes("misleading") || normalizedVerdict.includes("partial"))
      return <AlertTriangle className="w-5 h-5" />
    return <HelpCircle className="w-5 h-5" />
  }

  const getVerdictColor = (verdict: string) => {
    const normalizedVerdict = verdict?.toLowerCase() ?? ""
    if (normalizedVerdict.includes("true"))
      return "text-green-700 bg-green-50 border-green-200 dark:text-green-400 dark:bg-green-950 dark:border-green-800"
    if (normalizedVerdict.includes("false"))
      return "text-red-700 bg-red-50 border-red-200 dark:text-red-400 dark:bg-red-950 dark:border-red-800"
    if (normalizedVerdict.includes("misleading") || normalizedVerdict.includes("partial"))
      return "text-yellow-700 bg-yellow-50 border-yellow-200 dark:text-yellow-400 dark:bg-yellow-950 dark:border-yellow-800"
    return "text-gray-700 bg-gray-50 border-gray-200 dark:text-gray-400 dark:bg-gray-950 dark:border-gray-800"
  }

  const handleSubmit = async () => {
    setIsLoading(true)
    setResults(null)
    setReplies([])

    let endpoint = ""
    let options: RequestInit = {}

    try {
      if (activeTab === "file" && files && files.length > 0) {
        endpoint = `${API_BASE_URL}/fact-check-file`
        const formData = new FormData()
        formData.append("claim", claim)
        for (let i = 0; i < files.length; i++) {
          formData.append("files", files[i])
        }
        options = { method: "POST", body: formData }
      } else if (activeTab === "url" && url.trim()) {
        endpoint = `${API_BASE_URL}/fact-check-url`
        options = {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: url, claim: claim }),
        }
      } else if (activeTab === "text" && claim.trim()) {
        endpoint = `${API_BASE_URL}/fact-check-text`
        options = {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ claim: claim }),
        }
      } else {
        alert("Please provide input for the selected tab.")
        setIsLoading(false)
        return
      }

      const response = await fetch(endpoint, options)
      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.error || `HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      setResults(data)
    } catch (error: any) {
      console.error("Error during fact-check:", error)
      setResults({
        error: error.message || "Could not retrieve fact-check result. Please try again later.",
        claim_analysis: { verdict: "", score: 0, explanation: "" },
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleGenerateReply = async () => {
    if (!results) return
    setIsReplyLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/generate-reply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ analysis: results, language: "English" }),
      });
      if (!response.ok) {
        throw new Error("Failed to generate replies");
      }
      const data = await response.json();
      setReplies(data.replies || []);
    } catch (error) {
      console.error("Error generating replies:", error);
      alert("Could not generate replies. Please try again.");
    } finally {
      setIsReplyLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      <Navbar />
      <main className="flex-1">
        <div className="container mx-auto px-4 py-8 max-w-4xl">
          <header className="text-center mb-8 animate-in fade-in-0 slide-in-from-bottom-4 duration-700">
             <div className="flex justify-between items-start mb-6">
               <div className="flex-1" />
               <div className="flex-1 text-center">
                   <h1 className="text-4xl font-bold mb-2 text-balance">Fact-Checker</h1>
               </div>
               <div className="flex-1 flex justify-end">
                   <ThemeToggle />
               </div>
            </div>
            <p className="text-lg text-muted-foreground text-pretty">
              Verify claims, URLs, or files using AI-powered analysis and real-time web search.
            </p>
          </header>

          <Card className="mb-8 animate-in fade-in-0 slide-in-from-bottom-4 duration-700 delay-150">
            <CardContent className="p-6">
              <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="text"><FileText className="w-4 h-4 mr-2" />Text</TabsTrigger>
                  <TabsTrigger value="url"><LinkIcon className="w-4 h-4 mr-2" />URL</TabsTrigger>
                  <TabsTrigger value="file"><Upload className="w-4 h-4 mr-2" />File</TabsTrigger>
                </TabsList>
                <div className="mt-4 space-y-4">
                  <div>
                    <Label htmlFor="claim-input" className="text-sm font-medium">
                      Claim or Context (Optional for URL/File)
                    </Label>
                    <Textarea
                      id="claim-input"
                      placeholder="e.g., 'Is this message about a new government scheme true?'"
                      value={claim}
                      onChange={(e) => setClaim(e.target.value)}
                      className="min-h-[80px] mt-2"
                    />
                  </div>
                  <TabsContent value="text" className="m-0"></TabsContent>
                  <TabsContent value="url" className="m-0">
                    <Label htmlFor="url-input" className="text-sm font-medium">Website URL</Label>
                    <Input id="url-input" type="url" placeholder="https://example.com/news/article" value={url} onChange={(e) => setUrl(e.target.value)} className="mt-2" />
                  </TabsContent>
                  <TabsContent value="file" className="m-0">
                    <Label htmlFor="file-input" className="text-sm font-medium">Upload Image or PDF</Label>
                    <Input id="file-input" type="file" accept="image/*,.pdf" multiple onChange={(e) => setFiles(e.target.files)} className="mt-2 cursor-pointer" />
                  </TabsContent>
                </div>
              </Tabs>
              <Button onClick={handleSubmit} disabled={isLoading} className="w-full mt-6" size="lg">
                {isLoading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Analyzing...</> : <><Search className="w-4 h-4 mr-2" />Check Claim</>}
              </Button>
            </CardContent>
          </Card>

          {(isLoading || results) && (
             <Card className="animate-in fade-in-0 slide-in-from-bottom-4 duration-700 delay-300">
               <CardContent className="p-6">
                 {isLoading && (
                   <div className="flex items-center justify-center py-8">
                     <div className="flex flex-col items-center gap-4">
                       <Loader2 className="w-8 h-8 animate-spin text-primary" />
                       <div className="text-sm text-muted-foreground animate-pulse">Analyzing with AI...</div>
                     </div>
                   </div>
                 )}
                 {results && !isLoading && (
                   <div className="space-y-6">
                     {results.error ? (
                       <div className="text-red-600 dark:text-red-400 p-4 bg-red-50 dark:bg-red-950 rounded-lg border border-red-200 dark:border-red-800">
                         <strong>Server Error:</strong> {results.error}
                       </div>
                     ) : (
                       <>
                         {results.claim_analysis?.verdict && (
                           <div className="animate-in fade-in-0 slide-in-from-left-4 duration-500">
                             <div className={`p-4 rounded-lg border flex items-center gap-3 ${getVerdictColor(results.claim_analysis.verdict)}`}>
                               {getVerdictIcon(results.claim_analysis.verdict)}
                               <div className="font-semibold text-lg">
                                 Verdict: {results.claim_analysis.verdict}
                                 <span className="ml-2 text-sm font-normal">(Score: {results.claim_analysis.score ?? "N/A"}/100)</span>
                               </div>
                             </div>
                             <Card className="mt-4"><CardHeader><CardTitle className="text-lg">Explanation</CardTitle></CardHeader><CardContent><p className="leading-relaxed whitespace-pre-wrap">{results.claim_analysis.explanation || "No explanation provided."}</p></CardContent></Card>
                           </div>
                         )}
                         
                         { (results.claim_analysis?.verdict?.toLowerCase().includes("false") || results.claim_analysis?.verdict?.toLowerCase().includes("misleading")) && (
                           <div className="animate-in fade-in-0 slide-in-from-left-4 duration-500 delay-50">
                             <Button onClick={handleGenerateReply} disabled={isReplyLoading} className="w-full">
                               {isReplyLoading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Generating Replies...</> : <><Send className="w-4 h-4 mr-2" />Generate a Reply to Share</>}
                             </Button>
                             {replies.length > 0 && (
                               <div className="mt-4 space-y-2">
                                 <h4 className="text-sm font-semibold">Suggested Replies (Click to copy):</h4>
                                 {replies.map((reply, index) => (
                                   <div key={index}
                                     className="p-3 bg-muted/50 rounded-md text-sm cursor-pointer hover:bg-muted"
                                     onClick={() => navigator.clipboard.writeText(reply)}
                                     title="Copy to clipboard"
                                   >
                                     <p>{reply}</p>
                                   </div>
                                 ))}
                               </div>
                             )}
                           </div>
                         )}

                         { (results.categorized_points?.points_supporting_truthfulness?.length > 0 || results.categorized_points?.points_refuting_the_claim?.length > 0) && (
                           <Card className="animate-in fade-in-0 slide-in-from-left-4 duration-500 delay-100">
                             <CardHeader><CardTitle className="text-lg">Key Points Analysis</CardTitle></CardHeader>
                             <CardContent className="space-y-4">
                               {results.categorized_points?.points_supporting_truthfulness?.length > 0 && (
                                 <div>
                                   <h4 className="font-semibold mb-2 flex items-center gap-2 text-green-600 dark:text-green-500"><ThumbsUp className="w-4 h-4"/>Points Supporting Truthfulness</h4>
                                   <ul className="space-y-2 pl-5 list-disc text-muted-foreground">
                                     {results.categorized_points.points_supporting_truthfulness.map((point, i) => <li key={i}>{point}</li>)}
                                   </ul>
                                 </div>
                               )}
                               {results.categorized_points?.points_refuting_the_claim?.length > 0 && (
                                 <div>
                                   <h4 className="font-semibold mb-2 flex items-center gap-2 text-red-600 dark:text-red-500"><ThumbsDown className="w-4 h-4"/>Points Refuting the Claim</h4>
                                   <ul className="space-y-2 pl-5 list-disc text-muted-foreground">
                                     {results.categorized_points.points_refuting_the_claim.map((point, i) => <li key={i}>{point}</li>)}
                                   </ul>
                                 </div>
                               )}
                             </CardContent>
                           </Card>
                         )}

                         {results.risk_assessment?.possible_consequences?.length > 0 && (
                           <Card className="animate-in fade-in-0 slide-in-from-left-4 duration-500 delay-200">
                             <CardHeader><CardTitle className="text-lg flex items-center gap-2"><ShieldAlert className="w-5 h-5"/>Risk Assessment</CardTitle></CardHeader>
                             <CardContent>
                               <ul className="space-y-2 pl-5 list-disc text-muted-foreground">
                                 {results.risk_assessment.possible_consequences.map((consequence, i) => <li key={i}>{consequence}</li>)}
                               </ul>
                             </CardContent>
                           </Card>
                         )}

                         {(results.public_guidance_and_resources?.tips_to_identify_similar_scams || results.public_guidance_and_resources?.official_government_resources) && (
                           <Card className="animate-in fade-in-0 slide-in-from-left-4 duration-500 delay-300">
                             <CardHeader><CardTitle className="text-lg flex items-center gap-2"><BookUser className="w-5 h-5"/>Public Guidance & Resources</CardTitle></CardHeader>
                             <CardContent className="space-y-4">
                               {results.public_guidance_and_resources?.tips_to_identify_similar_scams?.length > 0 && (
                                 <div>
                                   <h4 className="font-semibold mb-2">How to Spot Similar Misinformation</h4>
                                   <ul className="space-y-2 pl-5 list-disc text-muted-foreground">
                                     {results.public_guidance_and_resources.tips_to_identify_similar_scams.map((tip, i) => <li key={i}>{tip}</li>)}
                                   </ul>
                                 </div>
                               )}
                               {results.public_guidance_and_resources?.official_government_resources && (
                                 <div>
                                   <h4 className="font-semibold mb-2">Official Resources</h4>
                                   <div className="text-sm space-y-2 text-muted-foreground">
                                     {results.public_guidance_and_resources.official_government_resources.relevant_agency_website && (
                                       <p><strong>Website:</strong> <a href={results.public_guidance_and_resources.official_government_resources.relevant_agency_website} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">{results.public_guidance_and_resources.official_government_resources.relevant_agency_website}</a></p>
                                     )}
                                     {results.public_guidance_and_resources.official_government_resources.national_helpline_number && (
                                       <p><strong>Helpline:</strong> {results.public_guidance_and_resources.official_government_resources.national_helpline_number}</p>
                                     )}
                                   </div>
                                 </div>
                               )}
                             </CardContent>
                           </Card>
                         )}

                         {results.evidence_log?.external_sources?.length > 0 && (
                           <Card className="animate-in fade-in-0 slide-in-from-left-4 duration-500 delay-400">
                             <CardHeader><CardTitle className="text-lg flex items-center gap-2"><Link2 className="w-5 h-5"/>Evidence Log</CardTitle></CardHeader>
                             <CardContent>
                               <ul className="space-y-2">
                                 {results.evidence_log.external_sources.map((source, i) => (
                                   <li key={i} className="text-sm truncate">
                                     <a href={source.url} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">{source.source_name}</a>
                                   </li>
                                 ))}
                               </ul>
                             </CardContent>
                           </Card>
                         )}
                       </>
                     )}
                   </div>
                 )}
               </CardContent>
             </Card>
          )}
        </div>
      </main>
      <Footer />
    </div>
  )
}