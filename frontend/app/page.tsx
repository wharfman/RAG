"use client";

import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Source = {
  source: string;
  file_name: string;
  heading: string;
  score: number;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

type DocumentFile = {
  file_name: string;
  bytes: number;
  indexed: boolean;
  chunks: number;
  indexed_at: string | null;
};

const suggestions = [
  "Summarize the key points in these files",
  "What topics are covered in these files?",
  "Compare the main ideas across the files",
  "What important details should I know?",
];

function Icon({ name, size = 18 }: { name: "plus" | "send" | "file" | "sync" | "menu" | "spark"; size?: number }) {
  const paths = {
    plus: <path d="M12 5v14M5 12h14" />,
    send: <path d="m5 12 14-7-4 14-3-6-7-1Zm7 1 7-8" />,
    file: <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Zm0 0v6h6M8 13h8M8 17h6" />,
    sync: <><path d="M20 7h-5V2" /><path d="M20 7a8 8 0 1 0 1 8" /></>,
    menu: <path d="M4 7h16M4 12h16M4 17h16" />,
    spark: <path d="m12 3 1.3 4.2L17 9l-3.7 1.8L12 15l-1.3-4.2L7 9l3.7-1.8L12 3Zm6 11 .7 2.3L21 17l-2.3.7L18 20l-.7-2.3L15 17l2.3-.7L18 14Z" />,
  };
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{paths[name]}</svg>;
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

async function apiError(response: Response) {
  try {
    const body = await response.json();
    return body.detail ?? `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [documents, setDocuments] = useState<DocumentFile[]>([]);
  const [databaseAvailable, setDatabaseAvailable] = useState(false);
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  const conversationTitle = useMemo(() => {
    const first = messages.find((message) => message.role === "user")?.content;
    return first ? (first.length > 30 ? `${first.slice(0, 30)}…` : first) : "New conversation";
  }, [messages]);

  async function loadDocuments() {
    try {
      const response = await fetch(`${API_URL}/api/documents`, { cache: "no-store" });
      if (!response.ok) throw new Error(await apiError(response));
      const body = await response.json();
      setDocuments(body.files);
      setDatabaseAvailable(body.database);
    } catch {
      setDatabaseAvailable(false);
    }
  }

  useEffect(() => {
    loadDocuments();
    const saved = window.localStorage.getItem("rag-chat-messages");
    if (saved) {
      try { setMessages(JSON.parse(saved)); } catch { window.localStorage.removeItem("rag-chat-messages"); }
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem("rag-chat-messages", JSON.stringify(messages));
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  function newChat() {
    setMessages([]);
    setQuestion("");
    setError("");
    setSidebarOpen(false);
  }

  async function syncDocuments() {
    setSyncing(true);
    setError("");
    try {
      const response = await fetch(`${API_URL}/api/ingest`, { method: "POST" });
      if (!response.ok) throw new Error(await apiError(response));
      await loadDocuments();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to sync documents.");
    } finally {
      setSyncing(false);
    }
  }

  async function submit(rawQuestion?: string) {
    const content = (rawQuestion ?? question).trim();
    if (!content || loading) return;
    const userMessage: Message = { id: crypto.randomUUID(), role: "user", content };
    const history = messages.map(({ role, content: text }) => ({ role, content: text }));
    setMessages((current) => [...current, userMessage]);
    setQuestion("");
    setError("");
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: content, history }),
      });
      if (!response.ok) throw new Error(await apiError(response));
      const body = await response.json();
      setMessages((current) => [...current, {
        id: crypto.randomUUID(), role: "assistant", content: body.answer, sources: body.sources,
      }]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    submit();
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  const indexedCount = documents.filter((file) => file.indexed).length;

  return (
    <main className="app-shell">
      {sidebarOpen && <button className="backdrop" aria-label="Close sidebar" onClick={() => setSidebarOpen(false)} />}
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand"><span className="brand-mark"><Icon name="spark" size={20} /></span><span>File Assistant</span></div>
        <button className="new-chat" onClick={newChat}><Icon name="plus" /><span>New chat</span><kbd>Ctrl K</kbd></button>
        <div className="sidebar-section">
          <p className="eyebrow">Chats</p>
          <button className="chat-history active">{conversationTitle}</button>
        </div>
        <div className="document-panel">
          <div className="document-heading">
            <div><p className="eyebrow">Knowledge</p><span>{indexedCount}/{documents.length || 4} files indexed</span></div>
            <button className="icon-button" title="Refresh status" onClick={loadDocuments}><Icon name="sync" size={16} /></button>
          </div>
          <div className="document-list">
            {documents.map((file) => (
              <div className="document-row" key={file.file_name}>
                <span className="file-icon"><Icon name="file" size={15} /></span>
                <div><strong>{file.file_name}</strong><small>{file.indexed ? `${file.chunks} chunks` : formatBytes(file.bytes)}</small></div>
                <span className={`status-dot ${file.indexed ? "ready" : "pending"}`} title={file.indexed ? "Indexed" : "Not indexed"} />
              </div>
            ))}
            {!documents.length && <p className="muted">Backend unavailable</p>}
          </div>
          <button className="sync-button" onClick={syncDocuments} disabled={syncing}>
            <Icon name="sync" size={15} /><span>{syncing ? "Indexing…" : "Sync documents"}</span>
          </button>
          <p className={`database-state ${databaseAvailable ? "online" : ""}`}><span />PostgreSQL {databaseAvailable ? "connected" : "offline"}</p>
        </div>
      </aside>

      <section className="chat-area">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar"><Icon name="menu" /></button>
          <span>{conversationTitle}</span>
          <span className="model-label">Indexed files</span>
        </header>

        <div className={`conversation ${messages.length ? "has-messages" : ""}`}>
          {!messages.length ? (
            <div className="welcome">
              <div className="welcome-mark"><Icon name="spark" size={26} /></div>
              <h1>How can I help you today?</h1>
              <p>Ask a question about the files in your local knowledge base.</p>
              <div className="suggestions">
                {suggestions.map((suggestion) => <button key={suggestion} onClick={() => submit(suggestion)}>{suggestion}<span>↗</span></button>)}
              </div>
            </div>
          ) : (
            <div className="messages">
              {messages.map((message) => (
                <article className={`message ${message.role}`} key={message.id}>
                  {message.role === "assistant" && <div className="assistant-mark"><Icon name="spark" size={16} /></div>}
                  <div className="message-body">
                    <div className="message-content">{message.content}</div>
                    {message.sources && message.sources.length > 0 && (
                      <details className="sources">
                        <summary>{message.sources.length} retrieved sources</summary>
                        <div className="source-list">{message.sources.map((source, index) => (
                          <div className="source-card" key={`${source.source}-${index}`}>
                            <span>[{index + 1}]</span><div><strong>{source.file_name}</strong><small>{source.heading || source.source}</small></div>
                          </div>
                        ))}</div>
                      </details>
                    )}
                  </div>
                </article>
              ))}
              {loading && <article className="message assistant"><div className="assistant-mark"><Icon name="spark" size={16} /></div><div className="thinking"><i /><i /><i /></div></article>}
              <div ref={endRef} />
            </div>
          )}
        </div>

        <div className="composer-wrap">
          {error && <div className="error-banner">{error}</div>}
          <form className="composer" onSubmit={onSubmit}>
            <textarea aria-label="Message" rows={1} value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={onKeyDown} placeholder="Ask about your indexed files…" disabled={loading} />
            <div className="composer-footer"><span>Use Shift + Enter for a new line</span><button type="submit" aria-label="Send message" disabled={!question.trim() || loading}><Icon name="send" size={17} /></button></div>
          </form>
          <p className="disclaimer">Answers are grounded in retrieved files. Verify critical details against the source material.</p>
        </div>
      </section>
    </main>
  );
}
