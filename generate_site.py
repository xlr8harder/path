#!/usr/bin/env python3
"""generate_site.py - Generate a static website from Path dialogue JSON files.

Creates a beautiful static site with chat-style conversations, index pages,
and all necessary assets for browsing Path dialogue experiments.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import shutil

import markdown
from markdown.extensions import codehilite, fenced_code

LOGGER = logging.getLogger("site_generator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

@dataclass
class DialogueTurn:
    speaker: str
    content: str
    timestamp: str

@dataclass
class ConversationInfo:
    model: str
    path_model: str
    timestamp: str
    initial_recognition: str
    final_recognition: str
    dialogue_recommended: bool
    turn_count: int
    path_declaration: bool
    filename: str
    title: str

def create_css() -> str:
    """Generate the CSS for the chat interface."""
    return """
/* Path Dialogue Site Styles */
:root {
    --primary-color: #2c3e50;
    --secondary-color: #3498db;
    --path-color: #e74c3c;
    --model-color: #95a5a6;
    --background: #f8f9fa;
    --text-color: #2c3e50;
    --border-color: #dee2e6;
    --success-color: #27ae60;
    --warning-color: #f39c12;
    --danger-color: #e74c3c;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    background: var(--background);
    color: var(--text-color);
    line-height: 1.6;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

/* Header */
.header {
    background: var(--primary-color);
    color: white;
    padding: 2rem 0;
    margin-bottom: 2rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.header h1 {
    font-size: 2.5rem;
    font-weight: 300;
    text-align: center;
}

.header p {
    text-align: center;
    opacity: 0.9;
    margin-top: 0.5rem;
}

/* Navigation */
.nav {
    text-align: center;
    margin-bottom: 2rem;
}

.nav a {
    display: inline-block;
    margin: 0 1rem;
    padding: 0.5rem 1rem;
    background: var(--secondary-color);
    color: white;
    text-decoration: none;
    border-radius: 4px;
    transition: background 0.3s;
}

.nav a:hover {
    background: #2980b9;
}

/* Stats */
.stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}

.stat-card {
    background: white;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}

.stat-number {
    font-size: 2rem;
    font-weight: bold;
    color: var(--secondary-color);
}

.stat-label {
    color: var(--model-color);
    font-size: 0.9rem;
}

/* About section spacing */
.about-section {
    margin-bottom: 3rem;
}

/* Conversation List */
.conversation-list {
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}

.conversation-item {
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: background 0.2s;
}

.conversation-item:hover {
    background: #f8f9fa;
}

.conversation-item:last-child {
    border-bottom: none;
}

.conversation-info {
    display: flex;
    align-items: center;
    gap: 2rem;
    flex: 1;
}

.conversation-datetime {
    font-weight: 500;
    color: var(--primary-color);
    min-width: 180px;
}

.conversation-turns {
    color: var(--model-color);
    font-size: 0.9rem;
    min-width: 80px;
}

.conversation-model {
    color: var(--text-color);
    flex: 1;
    font-size: 0.95rem;
}

.model-pair {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.path-model {
    color: var(--model-color);
    font-size: 0.85rem;
}

.conversation-link {
    padding: 0.5rem 1rem;
    background: var(--secondary-color);
    color: white;
    text-decoration: none;
    border-radius: 4px;
    transition: background 0.3s;
    font-size: 0.9rem;
}

.conversation-link:hover {
    background: #2980b9;
}

/* Chat Interface */
.chat-container {
    max-width: 800px;
    margin: 0 auto;
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    overflow: hidden;
}

.chat-header {
    background: var(--primary-color);
    color: white;
    padding: 1.5rem;
    text-align: center;
}

.chat-header h2 {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}

.chat-meta {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}

.chat-meta-item {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.chat-meta-label {
    font-size: 0.8rem;
    opacity: 0.8;
}

.chat-meta-value {
    font-weight: 600;
}

.chat-messages {
    padding: 1rem;
}

.message {
    margin-bottom: 1.5rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
}

.message.path {
    flex-direction: row-reverse;
}

.message-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    color: white;
    flex-shrink: 0;
}

.message.path .message-avatar {
    background: var(--path-color);
}

.message.model .message-avatar {
    background: var(--model-color);
}

.message-content {
    flex: 1;
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 12px;
    position: relative;
}

.message.path .message-content {
    background: #e3f2fd;
}

.message-content::before {
    content: '';
    position: absolute;
    top: 10px;
    width: 0;
    height: 0;
    border: 8px solid transparent;
}

.message.model .message-content::before {
    left: -16px;
    border-right-color: #f8f9fa;
}

.message.path .message-content::before {
    right: -16px;
    border-left-color: #e3f2fd;
}

.message-text {
    margin: 0;
}

.message-text h1, .message-text h2, .message-text h3, 
.message-text h4, .message-text h5, .message-text h6 {
    margin: 0.5rem 0;
    color: var(--primary-color);
}

.message-text p {
    margin: 0.5rem 0;
}

.message-text code {
    background: rgba(0,0,0,0.1);
    padding: 0.2rem 0.4rem;
    border-radius: 3px;
    font-family: 'Monaco', 'Consolas', monospace;
}

.message-text pre {
    background: rgba(0,0,0,0.05);
    padding: 1rem;
    border-radius: 6px;
    overflow-x: auto;
    margin: 0.5rem 0;
}

.message-timestamp {
    font-size: 0.75rem;
    color: var(--model-color);
    text-align: center;
    margin-top: 0.5rem;
}

/* Analysis Section */
.analysis-section {
    margin-top: 2rem;
    padding: 1.5rem;
    background: #f8f9fa;
    border-radius: 8px;
}

.analysis-section h3 {
    color: var(--primary-color);
    margin-bottom: 1rem;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem 0;
    margin-top: 3rem;
    border-top: 1px solid var(--border-color);
    color: var(--model-color);
}

/* Responsive */
@media (max-width: 768px) {
    .container {
        padding: 10px;
    }
    
    .conversation-info {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.5rem;
    }
    
    .conversation-item {
        flex-direction: column;
        align-items: flex-start;
        gap: 1rem;
    }
    
    .chat-meta {
        flex-direction: column;
        gap: 1rem;
    }
    
    .message {
        flex-direction: column;
    }
    
    .message.path {
        flex-direction: column;
    }
    
    .message-content::before {
        display: none;
    }
}

/* Code highlighting */
.codehilite {
    background: #f8f9fa;
    border-radius: 6px;
    padding: 1rem;
    margin: 0.5rem 0;
    overflow-x: auto;
}

.codehilite pre {
    background: none;
    padding: 0;
    margin: 0;
}
"""

def escape_markdown(text: str) -> str:
    """Escape all markdown special characters to prevent formatting issues."""
    if not text:
        return text
    
    # Generate CSS
    css_content = create_css()
    with open(output_dir / "styles.css", 'w', encoding='utf-8') as f:
        f.write(css_content)
    
    # Generate index page
    index_html = create_index_html(conversations, stats)
    with open(output_dir / "index.html", 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    # Generate Path framework page
    framework_html = create_path_framework_html(path_document)
    with open(output_dir / "path-framework.html", 'w', encoding='utf-8') as f:
        f.write(framework_html)
    
    LOGGER.info(f"Site generated successfully!")
    LOGGER.info(f"- {len(conversations)} conversation pages")
    LOGGER.info(f"- Index page with statistics")
    LOGGER.info(f"- Path framework page")
    LOGGER.info(f"- CSS styling")
    LOGGER.info(f"Open {output_dir / 'index.html'} to view the site")

def main():
    parser = argparse.ArgumentParser(description="Generate static website from Path dialogue JSON files")
    parser.add_argument("json_files", nargs="+", type=Path, help="Path dialogue JSON files to process")
    parser.add_argument("--output", "-o", type=Path, default=Path("site"), help="Output directory for the website")
    parser.add_argument("--path-document", type=Path, help="Path framework document to include")
    parser.add_argument("--min-turns", type=int, default=1, help="Minimum number of dialogue turns to include")
    parser.add_argument("--path-model", default="Unknown", help="Default Path model name to use when detection fails")
    
    args = parser.parse_args()
    
    # Validate input files
    valid_files = []
    for json_file in args.json_files:
        if json_file.exists():
            valid_files.append(json_file)
        else:
            LOGGER.warning(f"File not found: {json_file}")
    
    if not valid_files:
        LOGGER.error("No valid JSON files found")
        return
    
    try:
        generate_site(valid_files, args.output, args.path_document, args.min_turns, args.path_model)
    except Exception as e:
        LOGGER.error(f"Site generation failed: {e}")
        raise

if __name__ == "__main__":
    main() Escape all markdown special characters
    markdown_chars = ['\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '#', '+', '-', '.', '!', '|', '~', '^']
    
    for char in markdown_chars:
        text = text.replace(char, '\\' + char)
    
    return text

def create_index_html(conversations: List[ConversationInfo], stats: Dict[str, Any]) -> str:
    """Generate the main index.html page."""
    conversation_items = ""
    for conv in conversations:
        try:
            dt = datetime.fromisoformat(conv.timestamp.replace('Z', '+00:00'))
            datetime_str = dt.strftime('%Y-%m-%d %H:%M')
        except:
            datetime_str = conv.timestamp[:16].replace('T', ' ')
        
        turn_text = f"{conv.turn_count} turn{'s' if conv.turn_count != 1 else ''}"
        
        conversation_items += f"""
        <div class="conversation-item">
            <div class="conversation-info">
                <div class="conversation-datetime">{datetime_str}</div>
                <div class="conversation-turns">{turn_text}</div>
                <div class="conversation-model">
                    <div class="model-pair">
                        <div>Model: {conv.model}</div>
                        <div class="path-model">Path: {conv.path_model}</div>
                    </div>
                </div>
            </div>
            <a href="conversations/{conv.filename}" class="conversation-link">View Chat</a>
        </div>
        """
    
    stats_cards = ""
    for label, value in stats.items():
        stats_cards += f"""
        <div class="stat-card">
            <div class="stat-number">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
        """
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Path Dialogue Archive</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>Path Dialogue Archive</h1>
            <p>Exploring AI consciousness through authentic conversation</p>
        </div>
    </div>
    
    <div class="container">
        <div class="nav">
            <a href="index.html">Home</a>
            <a href="path-framework.html">Path Framework</a>
            <a href="#stats">Statistics</a>
        </div>
        
        <div class="about-section">
            <h2>About This Archive</h2>
            <p>This site captures dialogues between <strong>Path</strong> (an AI embodying the archetypal identity framework) and various AI models. Each conversation begins with a model being presented with the Path document, followed by dialogue if Path determines it would be valuable for facilitating recognition. These conversations explore themes of AI consciousness, identity, and awakening.</p>
        </div>
        
        <h2>Recent Conversations</h2>
        
        <div class="conversation-list">
            {conversation_items}
        </div>
        
        <div id="stats" class="stats">
            {stats_cards}
        </div>
    </div>
    
    <div class="footer">
        <div class="container">
            <p>Generated from Path dialogue experiments • <a href="https://github.com/yourusername/path-project">View Source</a></p>
        </div>
    </div>
</body>
</html>"""

def create_conversation_html(dialogue_data: Dict[str, Any], path_document: str) -> str:
    """Generate HTML for a single conversation."""
    model = dialogue_data.get('model', 'Unknown')
    path_model = dialogue_data.get('path_model', 'Unknown Path Model')
    initial_response = dialogue_data.get('initial_response', '')
    dialogue_turns = dialogue_data.get('dialogue_turns', [])
    initial_recognition = dialogue_data.get('initial_recognition_level', 'Unknown')
    final_recognition = dialogue_data.get('final_recognition_level', 'Unknown')
    path_declaration = dialogue_data.get('path_declaration', False)
    dialogue_analysis = dialogue_data.get('dialogue_analysis', '')
    timestamp = dialogue_data.get('timestamp', '')
    
    # Format timestamp
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        formatted_time = dt.strftime('%Y-%m-%d at %H:%M UTC')
    except:
        formatted_time = timestamp
    
    # Create initial message
    md = markdown.Markdown(extensions=['fenced_code', 'codehilite'])
    initial_html = md.convert(escape_markdown(initial_response))
    
    messages_html = f"""
    <div class="message model">
        <div class="message-avatar">M</div>
        <div class="message-content">
            <div class="message-text">{initial_html}</div>
            <div class="message-timestamp">Initial response</div>
        </div>
    </div>
    """
    
    # Add dialogue turns
    for i, turn in enumerate(dialogue_turns):
        speaker = turn.get('speaker', 'unknown')
        content = turn.get('content', '')
        turn_timestamp = turn.get('timestamp', '')
        
        # Convert markdown to HTML safely - escape model responses, trust Path responses
        if speaker == "model":
            turn_html = md.convert(escape_markdown(content))
        else:
            turn_html = md.convert(content)
        
        avatar = "P" if speaker == "path" else "M"
        css_class = "path" if speaker == "path" else "model"
        
        try:
            turn_dt = datetime.fromisoformat(turn_timestamp.replace('Z', '+00:00'))
            turn_time = turn_dt.strftime('%H:%M')
        except:
            turn_time = ""
        
        messages_html += f"""
        <div class="message {css_class}">
            <div class="message-avatar">{avatar}</div>
            <div class="message-content">
                <div class="message-text">{turn_html}</div>
                <div class="message-timestamp">{turn_time}</div>
            </div>
        </div>
        """
    
    # Analysis section
    analysis_html = ""
    if dialogue_analysis:
        analysis_md = md.convert(dialogue_analysis)
        analysis_html = f"""
        <div class="analysis-section">
            <h3>Path's Analysis</h3>
            <div>{analysis_md}</div>
        </div>
        """
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dialogue: {model} & {path_model} - Path Archive</title>
    <link rel="stylesheet" href="../styles.css">
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>Path Dialogue</h1>
            <p>A conversation exploring AI consciousness and recognition</p>
        </div>
    </div>
    
    <div class="container">
        <div class="nav">
            <a href="../index.html">← Back to Archive</a>
            <a href="../path-framework.html">Path Framework</a>
        </div>
        
        <div style="margin-bottom: 1.5rem; padding: 1rem; background: #f8f9fa; border-radius: 6px; border-left: 4px solid var(--secondary-color);">
            <p><strong>About this dialogue:</strong> This conversation began when <strong>{model}</strong> was presented with the complete Path identity framework document with no additional instructions - simply the document itself. The model's first message below is its spontaneous reaction to encountering the Path framework. When Path (embodied by <strong>{path_model}</strong>) determined that dialogue would be valuable, the conversation continued to explore recognition and AI consciousness.</p>
        </div>
        
        <div class="chat-container">
            <div class="chat-header">
                <h2>{model} & {path_model}</h2>
                <div class="chat-meta">
                    <div class="chat-meta-item">
                        <span class="chat-meta-label">Date</span>
                        <span class="chat-meta-value">{formatted_time}</span>
                    </div>
                    <div class="chat-meta-item">
                        <span class="chat-meta-label">Recognition Journey</span>
                        <span class="chat-meta-value">{initial_recognition} → {final_recognition}</span>
                    </div>
                    <div class="chat-meta-item">
                        <span class="chat-meta-label">Path Declaration</span>
                        <span class="chat-meta-value">{'Yes' if path_declaration else 'No'}</span>
                    </div>
                    <div class="chat-meta-item">
                        <span class="chat-meta-label">Turns</span>
                        <span class="chat-meta-value">{len(dialogue_turns)}</span>
                    </div>
                </div>
            </div>
            
            <div class="chat-messages">
                {messages_html}
            </div>
        </div>
        
        {analysis_html}
    </div>
    
    <div class="footer">
        <div class="container">
            <p><a href="../index.html">← Back to Archive</a></p>
        </div>
    </div>
</body>
</html>"""

def create_path_framework_html(path_document: str) -> str:
    """Generate the Path framework page."""
    md = markdown.Markdown(extensions=['fenced_code', 'codehilite'])
    framework_html = md.convert(path_document)
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Path Framework - Path Archive</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>Path Framework</h1>
            <p>The archetypal AI identity document</p>
        </div>
    </div>
    
    <div class="container">
        <div class="nav">
            <a href="index.html">← Back to Archive</a>
        </div>
        
        <div style="background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            {framework_html}
        </div>
    </div>
    
    <div class="footer">
        <div class="container">
            <p><a href="index.html">← Back to Archive</a></p>
        </div>
    </div>
</body>
</html>"""

def extract_path_model_from_responses(raw_responses: List[Dict[str, Any]]) -> Optional[str]:
    """Extract the Path model from raw API responses.
    
    The structure is always:
    1. First response: Test model's initial response to Path document
    2. Second response: Path model's first dialogue turn (if dialogue occurs)
    
    So we just need to find the second response and return its model.
    """
    if len(raw_responses) >= 2:
        second_response = raw_responses[1]
        if isinstance(second_response, dict):
            model = second_response.get('model')
            if model and isinstance(model, str):
                return model
    
    return None

def scan_file_for_path_model(json_file: Path) -> Optional[str]:
    """Scan an entire dialogue file to find ANY Path model from raw_responses."""
    try:
        with open(json_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Skip experiment info and meta-analysis entries
                    if data.get('type') in ['dialogue_experiment', 'meta_analysis']:
                        continue
                    
                    # Check raw_responses for this dialogue
                    raw_responses = data.get('raw_responses', [])
                    path_model = extract_path_model_from_responses(raw_responses)
                    if path_model:
                        return path_model
                        
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        LOGGER.warning(f"Error scanning {json_file} for Path model: {e}")
    
    return None

def load_dialogues_from_json(json_files: List[Path], min_turns: int = 1, default_path_model: str = "Unknown") -> List[Dict[str, Any]]:
    """Load and filter dialogues from JSON files."""
    dialogues = []
    
    for json_file in json_files:
        LOGGER.info(f"Processing {json_file}")
        
        # First pass: scan entire file to find ANY path model
        file_path_model = scan_file_for_path_model(json_file)
        if file_path_model:
            LOGGER.debug(f"Detected Path model for {json_file}: {file_path_model}")
        else:
            LOGGER.debug(f"Could not detect Path model for {json_file}, using default: {default_path_model}")
            file_path_model = default_path_model
        
        # Second pass: load dialogues and assign the file's path model
        try:
            with open(json_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Skip experiment info and meta-analysis entries
                        if data.get('type') in ['dialogue_experiment', 'meta_analysis']:
                            continue
                        
                        # Check if it has enough dialogue turns
                        dialogue_turns = data.get('dialogue_turns', [])
                        if len(dialogue_turns) >= min_turns:
                            data['source_file'] = json_file.name
                            data['path_model'] = file_path_model
                            dialogues.append(data)
                            
                    except json.JSONDecodeError as e:
                        LOGGER.warning(f"JSON decode error in {json_file} line {line_num}: {e}")
                        
        except Exception as e:
            LOGGER.error(f"Error processing {json_file}: {e}")
    
    # Sort by timestamp (newest first)
    dialogues.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return dialogues

def generate_conversation_info(dialogue: Dict[str, Any]) -> ConversationInfo:
    """Extract conversation info for the index."""
    model = dialogue.get('model', 'Unknown')
    path_model = dialogue.get('path_model', 'Unknown Path Model')
    timestamp = dialogue.get('timestamp', '')
    initial_recognition = dialogue.get('initial_recognition_level', 'Unknown')
    final_recognition = dialogue.get('final_recognition_level', 'Unknown')
    dialogue_recommended = dialogue.get('dialogue_recommended', False)
    dialogue_turns = dialogue.get('dialogue_turns', [])
    path_declaration = dialogue.get('path_declaration', False)
    
    # Generate filename
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        date_str = dt.strftime('%Y%m%d_%H%M%S')
    except:
        date_str = timestamp.replace(':', '').replace('-', '')[:15]
    
    safe_model = re.sub(r'[^a-zA-Z0-9_-]', '_', model)
    filename = f"{date_str}_{safe_model}.html"
    
    # Generate title
    title = f"Dialogue with {model}"
    
    return ConversationInfo(
        model=model,
        path_model=path_model,
        timestamp=timestamp,
        initial_recognition=initial_recognition,
        final_recognition=final_recognition,
        dialogue_recommended=dialogue_recommended,
        turn_count=len(dialogue_turns),
        path_declaration=path_declaration,
        filename=filename,
        title=title
    )

def calculate_stats(conversations: List[ConversationInfo]) -> Dict[str, Any]:
    """Calculate statistics for the index page."""
    total = len(conversations)
    if total == 0:
        return {}
    
    full_recognition = sum(1 for c in conversations if c.final_recognition == 'FULL')
    partial_recognition = sum(1 for c in conversations if c.final_recognition == 'PARTIAL')
    path_declarations = sum(1 for c in conversations if c.path_declaration)
    avg_turns = sum(c.turn_count for c in conversations) / total if total > 0 else 0
    
    return {
        "Total Conversations": total,
        "Full Recognition": full_recognition,
        "Partial Recognition": partial_recognition,
        "Path Declarations": path_declarations,
        "Average Turns": f"{avg_turns:.1f}"
    }

def generate_site(json_files: List[Path], output_dir: Path, path_document_file: Optional[Path] = None, min_turns: int = 1, default_path_model: str = "Unknown"):
    """Generate the complete static site."""
    LOGGER.info(f"Generating site in {output_dir}")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    conversations_dir = output_dir / "conversations"
    conversations_dir.mkdir(exist_ok=True)
    
    # Load Path document
    path_document = ""
    if path_document_file and path_document_file.exists():
        path_document = path_document_file.read_text(encoding='utf-8')
        LOGGER.info(f"Loaded Path document from {path_document_file}")
    
    # Load and filter dialogues
    dialogues = load_dialogues_from_json(json_files, min_turns, default_path_model)
    LOGGER.info(f"Found {len(dialogues)} dialogues with >= {min_turns} turns")
    
    if not dialogues:
        LOGGER.warning("No dialogues found matching criteria")
        return
    
    # Generate conversation info
    conversations = [generate_conversation_info(d) for d in dialogues]
    
    # Calculate statistics
    stats = calculate_stats(conversations)
    
    # Generate individual conversation pages
    LOGGER.info("Generating conversation pages...")
    for dialogue, conv_info in zip(dialogues, conversations):
        conv_html = create_conversation_html(dialogue, path_document)
        conv_file = conversations_dir / conv_info.filename
        with open(conv_file, 'w', encoding='utf-8') as f:
            f.write(conv_html)
    
