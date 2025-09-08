import re
from typing import Dict, List, Any, Optional, Tuple
import json

class DecisionEngine:
    def __init__(self):
        self.decision_patterns = {
            "vs_comparison": [r"(\w+)\s+vs\s+(\w+)", r"(\w+)\s+or\s+(\w+)", r"compare\s+(\w+)\s+and\s+(\w+)"],
            "binary_choice": [r"should\s+i", r"which\s+is\s+better", r"recommend", r"choose"],
            "prediction": [r"will\s+(\w+)\s+win", r"who\s+will\s+win", r"predict", r"forecast"],
            "investment": [r"buy\s+or\s+sell", r"invest\s+in", r"stock\s+pick"]
        }
    
    def extract_decision_context(self, query: str) -> Dict[str, Any]:
        """Extract decision context from query"""
        query_lower = query.lower()
        
        # Find decision type
        decision_type = "analysis"
        options = []
        
        for pattern_type, patterns in self.decision_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    decision_type = pattern_type
                    if match.groups():
                        options.extend([g.strip() for g in match.groups() if g])
                    break
        
        # Extract specific options
        if not options:
            # Look for common comparison words
            comparison_words = ["vs", "versus", "or", "between"]
            for word in comparison_words:
                if word in query_lower:
                    parts = query_lower.split(word)
                    if len(parts) >= 2:
                        # Extract potential options around comparison word
                        before = parts[0].split()[-3:] if parts[0].split() else []
                        after = parts[1].split()[:3] if parts[1].split() else []
                        options.extend([w for w in before + after if len(w) > 2])
        
        return {
            "decision_type": decision_type,
            "options": list(set(options))[:4],  # Max 4 options
            "requires_decision": decision_type != "analysis",
            "binary_choice": len(options) == 2
        }
    
    def generate_decision(self, analysis_result: str, context: Dict[str, Any], confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """Generate final decision based on analysis"""
        
        if not context.get("requires_decision"):
            return {"decision": None, "reasoning": "No decision required"}
        
        # Score options based on analysis content
        options = context.get("options", [])
        if not options:
            return {"decision": "Insufficient data", "reasoning": "No clear options identified"}
        
        option_scores = {}
        analysis_lower = analysis_result.lower()
        
        # Simple scoring based on positive/negative context
        positive_words = ["better", "stronger", "higher", "good", "excellent", "recommended", "positive", "growth", "win", "success"]
        negative_words = ["worse", "weaker", "lower", "bad", "poor", "avoid", "negative", "decline", "lose", "risk"]
        
        for option in options:
            option_lower = option.lower()
            score = 0
            
            # Count positive mentions
            for pos_word in positive_words:
                if f"{option_lower} {pos_word}" in analysis_lower or f"{pos_word} {option_lower}" in analysis_lower:
                    score += 2
                elif pos_word in analysis_lower and option_lower in analysis_lower:
                    score += 1
            
            # Count negative mentions
            for neg_word in negative_words:
                if f"{option_lower} {neg_word}" in analysis_lower or f"{neg_word} {option_lower}" in analysis_lower:
                    score -= 2
                elif neg_word in analysis_lower and option_lower in analysis_lower:
                    score -= 1
            
            # Frequency bonus
            option_mentions = analysis_lower.count(option_lower)
            score += min(option_mentions * 0.5, 3)  # Cap frequency bonus
            
            option_scores[option] = score
        
        # Determine winner
        if not option_scores:
            return {"decision": "Inconclusive", "reasoning": "Unable to score options"}
        
        sorted_options = sorted(option_scores.items(), key=lambda x: x[1], reverse=True)
        winner = sorted_options[0]
        runner_up = sorted_options[1] if len(sorted_options) > 1 else None
        
        # Calculate confidence
        max_score = winner[1]
        second_score = runner_up[1] if runner_up else 0
        score_diff = max_score - second_score
        confidence = min(0.95, max(0.5, (score_diff + 5) / 10))  # Normalize to 0.5-0.95
        
        # Generate decision
        decision_text = winner[0].title()
        
        if confidence >= confidence_threshold:
            recommendation = f"**RECOMMENDATION: {decision_text}**"
            confidence_level = "High"
        elif confidence >= 0.6:
            recommendation = f"**LEAN TOWARDS: {decision_text}**"
            confidence_level = "Medium"
        else:
            recommendation = "**INCONCLUSIVE** - More analysis needed"
            confidence_level = "Low"
        
        reasoning_parts = []
        if winner[1] > 0:
            reasoning_parts.append(f"{winner[0]} shows stronger positive indicators")
        if runner_up and runner_up[1] < 0:
            reasoning_parts.append(f"{runner_up[0]} has concerning negative factors")
        if score_diff > 3:
            reasoning_parts.append("Clear performance differential identified")
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Based on available analysis"
        
        return {
            "decision": recommendation,
            "winner": decision_text,
            "confidence": round(confidence, 2),
            "confidence_level": confidence_level,
            "reasoning": reasoning,
            "scores": option_scores,
            "all_options": sorted_options
        }
    
    def format_decision_output(self, decision_data: Dict[str, Any], query: str) -> str:
        """Format decision for display"""
        
        if not decision_data.get("decision"):
            return ""
        
        output = f"\n\n## 🎯 **FINAL DECISION**\n\n"
        output += f"### {decision_data['decision']}\n\n"
        
        if decision_data.get("confidence"):
            confidence_emoji = "🟢" if decision_data["confidence"] >= 0.7 else "🟡" if decision_data["confidence"] >= 0.6 else "🔴"
            output += f"**Confidence:** {confidence_emoji} {decision_data['confidence_level']} ({decision_data['confidence']:.0%})\n\n"
        
        if decision_data.get("reasoning"):
            output += f"**Key Reasoning:** {decision_data['reasoning']}\n\n"
        
        if decision_data.get("scores"):
            output += "**Option Scores:**\n"
            for option, score in decision_data["all_options"]:
                score_emoji = "📈" if score > 0 else "📉" if score < 0 else "➖"
                output += f"- {score_emoji} **{option.title()}**: {score:+.1f}\n"
        
        return output

# Integration function for main app
def enhance_with_decision(analysis_result: str, query: str) -> str:
    """Add decision output to analysis result"""
    
    engine = DecisionEngine()
    context = engine.extract_decision_context(query)
    
    if context.get("requires_decision"):
        decision_data = engine.generate_decision(analysis_result, context)
        decision_output = engine.format_decision_output(decision_data, query)
        return analysis_result + decision_output
    
    return analysis_result

# Export
__all__ = ['DecisionEngine', 'enhance_with_decision']