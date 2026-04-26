import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [tool, setTool] = useState('Flashcards');
  const [topic, setTopic] = useState('');
  const [files, setFiles] = useState([]);
  const [status, setStatus] = useState({ type: '', msg: '' });
  const [loading, setLoading] = useState(false);
  const [hasKB, setHasKB] = useState(false);
  const [result, setResult] = useState(null);
  
  // Quiz State
  const [quizAnswers, setQuizAnswers] = useState({});
  const [quizSubmitted, setQuizSubmitted] = useState(false);

  useEffect(() => {
    checkKBStatus();
  }, []);

  const checkKBStatus = async () => {
    try {
      const res = await axios.get(`${API_BASE}/status`);
      setHasKB(res.data.has_knowledge_base);
    } catch (e) {
      console.error(e);
    }
  };

  const handleFileChange = (e) => {
    setFiles(e.target.files);
  };

  const processFiles = async () => {
    if (files.length === 0) {
      setStatus({ type: 'error', msg: 'Please select PDF files first.' });
      return;
    }

    setLoading(true);
    setStatus({ type: '', msg: '' });
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    try {
      const res = await axios.post(`${API_BASE}/upload`, formData);
      setStatus({ type: 'success', msg: `✅ Processed ${files.length} file(s) → ${res.data.chunks} chunks embedded!` });
      setHasKB(true);
      setFiles([]);
    } catch (err) {
      setStatus({ type: 'error', msg: 'Failed to process files.' });
    }
    setLoading(false);
  };

  const clearKB = async () => {
    try {
      await axios.post(`${API_BASE}/clear`);
      setHasKB(false);
      setResult(null);
      setStatus({ type: 'success', msg: 'Knowledge base cleared! Upload new PDFs.' });
    } catch (e) {
      setStatus({ type: 'error', msg: 'Failed to clear KB.' });
    }
  };

  const generate = async () => {
    if (!topic) {
      setStatus({ type: 'error', msg: 'Please enter a topic.' });
      return;
    }
    if (!hasKB) {
      setStatus({ type: 'error', msg: '📭 No documents in the knowledge base. Please upload PDFs first.' });
      return;
    }

    setLoading(true);
    setResult(null);
    setStatus({ type: '', msg: '' });
    setQuizAnswers({});
    setQuizSubmitted(false);

    let endpoint = '';
    if (tool === 'Flashcards') endpoint = '/generate/flashcards';
    else if (tool === 'Planner') endpoint = '/generate/planner';
    else if (tool === 'Quiz') endpoint = '/generate/quiz';

    try {
      const res = await axios.post(`${API_BASE}${endpoint}`, { topic });
      let data = res.data.data;
      
      if (data && data.error) {
        setStatus({ type: 'error', msg: data.error });
      } else {
        setResult(data);
      }
    } catch (err) {
      setStatus({ type: 'error', msg: err.response?.data?.detail || 'Generation failed.' });
    }
    setLoading(false);
  };

  const renderFlashcards = () => {
    if (!Array.isArray(result)) return <p>Invalid format</p>;
    return (
      <div className="flashcards-grid">
        {result.map((card, i) => (
          <div key={i} className="flashcard">
            <h3>{card.term}</h3>
            <p className="definition">{card.definition}</p>
          </div>
        ))}
      </div>
    );
  };

  const renderPlanner = () => {
    return (
      <div className="markdown-body">
        <ReactMarkdown>{result}</ReactMarkdown>
      </div>
    );
  };

  const handleQuizChange = (i, val) => {
    if (!quizSubmitted) {
      setQuizAnswers(prev => ({ ...prev, [i]: val }));
    }
  };

  const renderQuiz = () => {
    if (!Array.isArray(result)) return <p>Invalid format</p>;
    
    let score = 0;
    const total = result.length;

    return (
      <div className="quiz-container">
        {result.map((q, i) => {
          const correct = q.correct?.trim()?.toUpperCase();
          const userAns = quizAnswers[i];
          const isCorrect = userAns === correct;
          if (quizSubmitted && isCorrect) score++;

          return (
            <div key={i} className="quiz-question">
              <h3>Q{i + 1}. {q.question}</h3>
              <div className="quiz-options">
                {Object.entries(q.options || {}).map(([key, val]) => (
                  <label key={key} className="quiz-option">
                    <input 
                      type="radio" 
                      name={`q-${i}`} 
                      value={key}
                      checked={userAns === key}
                      onChange={() => handleQuizChange(i, key)}
                      disabled={quizSubmitted}
                    />
                    <span>{key}. {val}</span>
                  </label>
                ))}
              </div>
              
              {quizSubmitted && (
                <div className="quiz-explanation">
                  <p><strong>{isCorrect ? '✅ Correct!' : `❌ Wrong — Correct was ${correct}`}</strong></p>
                  <p>{q.explanation}</p>
                  <small style={{ opacity: 0.7 }}>{q.citation}</small>
                </div>
              )}
            </div>
          );
        })}
        
        {!quizSubmitted ? (
          <button className="btn" onClick={() => setQuizSubmitted(true)}>📝 Submit Quiz</button>
        ) : (
          <div className="quiz-score">
            <h2>Score: {score} / {total} ({Math.round(score/total*100)}%)</h2>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="brand">
          <h1>Exam Assistant</h1>
          <p>AI-Powered Local Study Companion</p>
        </div>

        <div className="upload-section">
          <h3>📄 Upload Materials</h3>
          <div className="file-input-wrapper">
            <button className="btn btn-secondary">Select PDF Files ({files.length})</button>
            <input type="file" multiple accept=".pdf" onChange={handleFileChange} />
          </div>
          
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button className="btn" onClick={processFiles} disabled={loading}>
              {loading ? <div className="spinner"></div> : 'Process PDFs'}
            </button>
            <button className="btn btn-danger" onClick={clearKB}>🗑️ Clear KB</button>
          </div>
        </div>

        {status.msg && (
          <div className={`status-badge ${status.type}`}>
            {status.msg}
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className="tool-selector">
          {['Flashcards', 'Planner', 'Quiz'].map(t => (
            <button 
              key={t}
              className={`tool-btn ${tool === t ? 'active' : ''}`}
              onClick={() => { setTool(t); setResult(null); }}
            >
              {t === 'Flashcards' ? '🗂️ Flashcards' : t === 'Planner' ? '🗓️ Planner' : '🎯 Grounded Quiz'}
            </button>
          ))}
        </div>

        <div className="input-group">
          <input 
            type="text" 
            className="topic-input"
            placeholder="Enter the topic you want to focus on..."
            value={topic}
            onChange={e => setTopic(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && generate()}
          />
          <button className="btn generate-btn" onClick={generate} disabled={loading}>
            {loading ? <div className="spinner"></div> : 'Generate ✨'}
          </button>
        </div>

        {result && (
          <div className="results-area">
            {tool === 'Flashcards' && renderFlashcards()}
            {tool === 'Planner' && renderPlanner()}
            {tool === 'Quiz' && renderQuiz()}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
