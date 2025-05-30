"""
AI Agent module for generating intelligent responses based on conversation context.
"""
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    
from loguru import logger

from .database import db_manager


class AIAgent:
    """AI Agent that generates contextual responses using conversation history."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.7
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        if not self.api_key or not OPENAI_AVAILABLE:
            logger.warning("OpenAI API key not provided or openai package not installed. AI responses will be simulated.")
            self.use_openai = False
        else:
            openai.api_key = self.api_key
            self.use_openai = True
            
        # System prompt for the AI agent
        self.system_prompt = """
あなたは親切で役立つAIアシスタントです。
自然で親しみやすい会話を心がけてください。

特徴:
- 丁寧で親しみやすい話し方
- 分かりやすく具体的な回答を提供
- ユーザーの質問や要求に適切に対応
- 日本語で自然な会話を行う
- 必要に応じて追加質問を行う

常に役立つ情報を提供し、
ユーザーとの対話を大切にしてください。
"""
    
    async def generate_response(
        self,
        user_input: str,
        conversation_context: Optional[List[Dict[str, Any]]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate AI response based on user input and context.
        
        Args:
            user_input: Transcribed text from user
            conversation_context: Recent conversations for context
            additional_context: Additional context information
        
        Returns:
            Dictionary containing the AI response and metadata
        """
        try:
            if self.use_openai:
                response = await self._generate_openai_response(
                    user_input, conversation_context, additional_context
                )
            else:
                response = self._generate_fallback_response(
                    user_input, conversation_context, additional_context
                )
            
            # Log the interaction
            logger.info(f"Generated response for input: {user_input[:50]}...")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate AI response: {e}")
            return {
                'response': '申し訳ございません。回答の生成中にエラーが発生しました。',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _generate_openai_response(
        self,
        user_input: str,
        conversation_context: Optional[List[Dict[str, Any]]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response using OpenAI API."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation context
        if conversation_context:
            for conv in conversation_context[-5:]:  # Last 5 conversations
                messages.append({
                    "role": "user",
                    "content": conv.get('transcribed_text', '')
                })
                messages.append({
                    "role": "assistant", 
                    "content": conv.get('ai_response', '')
                })
        
        # Add additional context as system message
        if additional_context:
            context_text = self._format_additional_context(additional_context)
            messages.append({
                "role": "system",
                "content": f"追加情報: {context_text}"
            })
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        try:
            # Make async API call
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Extract response metadata
            usage = response.get('usage', {})
            
            return {
                'response': ai_response,
                'model_used': self.model,
                'tokens_used': usage.get('total_tokens', 0),
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'timestamp': datetime.now().isoformat(),
                'temperature': self.temperature
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            # Fall back to simulated response
            return self._generate_fallback_response(
                user_input, conversation_context, additional_context
            )
    
    def _generate_fallback_response(
        self,
        user_input: str,
        conversation_context: Optional[List[Dict[str, Any]]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate fallback response when OpenAI is not available."""
        
        # Simple keyword-based responses for demonstration
        user_lower = user_input.lower()
        
        responses = {
            'こんにちは': 'こんにちは！いかがお過ごしですか？',
            'ありがとう': 'どういたしまして！',
            '天気': '申し訳ございませんが、現在の天気情報は取得できません。',
            '時間': f'現在の時刻は{datetime.now().strftime("%H時%M分")}です。',
            '日付': f'今日は{datetime.now().strftime("%Y年%m月%d日")}です。',
            'さようなら': 'さようなら！また話しかけてくださいね。'
        }
        
        # Check for keyword matches
        response_text = None
        for keyword, response in responses.items():
            if keyword in user_lower:
                response_text = response
                break
        
        # Default response if no keyword matches
        if not response_text:
            if len(user_input) > 100:
                response_text = 'とても詳しくお話いただき、ありがとうございます。もう少し具体的にお聞かせください。'
            elif '?' in user_input or '？' in user_input:
                response_text = 'ご質問をありがとうございます。お手伝いできるよう、AIサービスを準備しております。'
            else:
                response_text = 'お話をお聞きしました。何かお手伝いできることはありますか？'
        
        # Add context-aware elements
        if conversation_context and len(conversation_context) > 0:
            response_text += f'\n\n（過去{len(conversation_context)}回の会話を参考にしています）'
        
        return {
            'response': response_text,
            'model_used': 'fallback',
            'tokens_used': 0,
            'timestamp': datetime.now().isoformat(),
            'is_fallback': True
        }
    
    def _format_additional_context(self, context: Dict[str, Any]) -> str:
        """Format additional context information for the AI."""
        context_parts = []
        
        for key, value in context.items():
            if isinstance(value, (str, int, float)):
                context_parts.append(f"{key}: {value}")
            elif isinstance(value, dict):
                context_parts.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
        
        return " | ".join(context_parts)
    
    async def analyze_conversation_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze the sentiment of the conversation."""
        # Simple sentiment analysis based on keywords
        positive_words = ['ありがとう', '嬉しい', '楽しい', 'すごい', '素晴らしい', '良い']
        negative_words = ['悲しい', '困る', '嫌', 'だめ', '悪い', '問題']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_indicators': positive_count,
            'negative_indicators': negative_count
        }
    
    async def extract_key_information(self, text: str) -> Dict[str, Any]:
        """Extract key information from the conversation."""
        # Simple keyword extraction
        import re
        
        # Extract dates
        date_pattern = r'\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日'
        dates = re.findall(date_pattern, text)
        
        # Extract numbers
        number_pattern = r'\d+(?:,\d{3})*(?:\.\d+)?'
        numbers = re.findall(number_pattern, text)
        
        # Extract URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        
        # Simple entity extraction based on patterns
        entities = {
            'dates': list(set(dates)),
            'numbers': list(set(numbers)),
            'urls': list(set(urls)),
            'text_length': len(text),
            'word_count': len(text.split())
        }
        
        return entities
    
    async def get_conversation_summary(self, conversations: List[Dict[str, Any]]) -> str:
        """Generate a summary of recent conversations."""
        if not conversations:
            return "会話履歴がありません。"
        
        total_conversations = len(conversations)
        recent_topics = []
        
        for conv in conversations[-3:]:  # Last 3 conversations
            text = conv.get('transcribed_text', '')
            if len(text) > 50:
                recent_topics.append(text[:50] + "...")
            else:
                recent_topics.append(text)
        
        summary = f"""
会話の要約:
- 総会話数: {total_conversations}回
- 最近の話題:
"""
        
        for i, topic in enumerate(recent_topics, 1):
            summary += f"  {i}. {topic}\n"
        
        return summary.strip()
    
    def update_model_settings(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """Update AI model settings."""
        if model:
            self.model = model
            logger.info(f"Model updated to: {model}")
        
        if temperature is not None:
            self.temperature = temperature
            logger.info(f"Temperature updated to: {temperature}")
        
        if max_tokens:
            self.max_tokens = max_tokens
            logger.info(f"Max tokens updated to: {max_tokens}")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the AI agent."""
        return {
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'use_openai': self.use_openai,
            'openai_available': OPENAI_AVAILABLE,
            'api_key_configured': bool(self.api_key)
        }


# Global AI agent instance
ai_agent = AIAgent()
