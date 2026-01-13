"""
Story Generation Library
A simple library for generating 5-page children's stories.

Main function: generate_story()
"""

import random
import re
from typing import Optional, Dict, List


# Age group configurations
AGE_CONFIGS = {
    "3-6": {
        "sentence_length": (5, 8),
        "total_words": (100, 150),
        "vocab_level": "kindergarten",
        "sentence_structure": "simple",
        "page_sentences": [2, 2, 3, 3, 2]
    },
    "7-10": {
        "sentence_length": (8, 12),
        "total_words": (200, 300),
        "vocab_level": "elementary",
        "sentence_structure": "compound",
        "page_sentences": [2, 3, 4, 4, 3]
    },
    "11-12": {
        "sentence_length": (12, 15),
        "total_words": (300, 400),
        "vocab_level": "middle_school",
        "sentence_structure": "complex",
        "page_sentences": [3, 3, 4, 4, 3]
    }
}

# Story elements
COMPANION_TYPES = [
    "a friendly guide", "a wise mentor", "a playful friend",
    "a helpful creature", "a magical helper"
]

def get_environment_details(story_world: str) -> str:
    """Get environment-specific details based on story world."""
    world_lower = story_world.lower()
    if 'enchanted forest' in world_lower or world_lower == 'forest':
        return "Include magical trees with glowing elements, mystical flora, enchanted atmosphere with soft magical light, and fairy-tale forest setting with whimsical details."
    elif 'outer space' in world_lower or world_lower == 'space':
        return "Include planets, stars, alien landscapes, cosmic scenery, space nebulas, celestial bodies, and otherworldly terrain."
    elif 'underwater kingdom' in world_lower or world_lower == 'underwater':
        return "Include coral reefs, sea creatures, underwater flora, aquatic plants, marine life, and oceanic elements."
    else:
        return "Match the setting and atmosphere of the story world."

CHALLENGES = {
    "3-6": [
        "find a lost treasure", "help a friend in need",
        "discover a secret path", "save a special place"
    ],
    "7-10": [
        "solve an ancient puzzle", "rescue someone in danger",
        "restore balance to the world", "uncover a hidden mystery"
    ],
    "11-12": [
        "face an inner fear", "make a difficult choice",
        "understand a complex truth", "transform a challenging situation"
    ]
}


def count_words(text: str) -> int:
    """Count the number of words in a text."""
    words = re.findall(r'\b\w+\b', text.lower())
    return len(words)


def create_simple_sentence(text: str, min_words: int, max_words: int) -> str:
    """Ensure sentence is within word count range."""
    words = text.split()
    if len(words) < min_words:
        additions = ["very", "so", "really", "too"]
        while len(words) < min_words and len(additions) > 0:
            words.insert(-1, random.choice(additions))
            additions.remove(words[-2]) if len(words) > 1 else None
    elif len(words) > max_words:
        words = words[:max_words]
    return " ".join(words).capitalize()


def generate_page_1(character_name: str, character_type: str, special_ability: str, age_group: str) -> str:
    """Generate Page 1: Character introduction with ability."""
    age_config = AGE_CONFIGS[age_group]
    num_sentences = age_config["page_sentences"][0]
    min_words, max_words = age_config["sentence_length"]
    
    sentences = []
    
    # Sentence 1: Introduce character and type
    if age_group == "3-6":
        s1 = f"{character_name} is {character_type}."
    elif age_group == "7-10":
        s1 = f"Meet {character_name}, {character_type} who loves adventures."
    else:
        s1 = f"In a world full of possibilities, {character_name} stands out as {character_type} with a unique spirit."
    
    sentences.append(create_simple_sentence(s1, min_words, max_words))
    
    # Sentence 2: Reveal special ability
    if age_group == "3-6":
        s2 = f"{character_name} can {special_ability}."
    elif age_group == "7-10":
        s2 = f"{character_name} has a special power: {character_name} can {special_ability}."
    else:
        s2 = f"What makes {character_name} extraordinary is the ability to {special_ability}, a gift that brings wonder and joy."
    
    sentences.append(create_simple_sentence(s2, min_words, max_words))
    
    # Sentence 3 (for older groups): Set positive tone
    if num_sentences >= 3:
        if age_group == "11-12":
            s3 = f"This ability fills {character_name} with confidence and excitement for what lies ahead."
            sentences.append(create_simple_sentence(s3, min_words, max_words))
    
    return " ".join(sentences) + " "


def generate_page_2(character_name: str, story_world: str, age_group: str) -> str:
    """Generate Page 2: Character enters world."""
    age_config = AGE_CONFIGS[age_group]
    num_sentences = age_config["page_sentences"][1]
    min_words, max_words = age_config["sentence_length"]
    
    sentences = []
    
    # Sentence 1: Discover portal/entrance
    if age_group == "3-6":
        s1 = f"One day {character_name} found a door to {story_world}."
    elif age_group == "7-10":
        s1 = f"While exploring, {character_name} discovered a magical entrance that led to {story_world}."
    else:
        s1 = f"During an ordinary moment, {character_name} stumbled upon a mysterious gateway that shimmered with possibility, revealing the path to {story_world}."
    
    sentences.append(create_simple_sentence(s1, min_words, max_words))
    
    # Sentence 2: First impressions
    if age_group == "3-6":
        s2 = f"{character_name} saw many wonderful things there."
    elif age_group == "7-10":
        s2 = f"As {character_name} stepped through, the world revealed itself with colors and sounds that filled {character_name} with wonder."
    else:
        s2 = f"Upon entering, {character_name} was immediately struck by the breathtaking beauty and the sense of adventure that permeated every corner of this new realm."
    
    sentences.append(create_simple_sentence(s2, min_words, max_words))
    
    # Sentence 3: What draws them in
    if num_sentences >= 3:
        if age_group == "3-6":
            s3 = f"{character_name} wanted to explore more."
        elif age_group == "7-10":
            s3 = f"Something inside {character_name} knew that an amazing adventure was about to begin."
        else:
            s3 = f"{character_name} felt a deep connection to this place and sensed that it held secrets waiting to be discovered."
        
        sentences.append(create_simple_sentence(s3, min_words, max_words))
    
    return " ".join(sentences) + " "


def generate_page_3(character_name: str, adventure_type: str, age_group: str) -> str:
    """Generate Page 3: Adventure begins."""
    age_config = AGE_CONFIGS[age_group]
    num_sentences = age_config["page_sentences"][2]
    min_words, max_words = age_config["sentence_length"]
    
    sentences = []
    challenge = random.choice(CHALLENGES[age_group])
    
    # Sentence 1: Adventure begins
    if age_group == "3-6":
        s1 = f"Then {character_name} started a {adventure_type}."
    elif age_group == "7-10":
        s1 = f"Suddenly, {character_name} realized that a {adventure_type} was beginning, and {character_name} was right in the middle of it."
    else:
        s1 = f"As {character_name} ventured deeper, it became clear that a {adventure_type} was unfolding, one that would test {character_name}'s resolve and character."
    
    sentences.append(create_simple_sentence(s1, min_words, max_words))
    
    # Sentence 2: Introduce challenge/quest
    if age_group == "3-6":
        s2 = f"{character_name} needed to {challenge}."
    elif age_group == "7-10":
        s2 = f"The mission was clear: {character_name} must {challenge}, but it wouldn't be easy."
    else:
        s2 = f"The challenge ahead required {character_name} to {challenge}, a task that would demand both courage and wisdom."
    
    sentences.append(create_simple_sentence(s2, min_words, max_words))
    
    # Sentence 3: Optional companion
    if num_sentences >= 3:
        companion = random.choice(COMPANION_TYPES)
        if age_group == "3-6":
            s3 = f"{character_name} met {companion} who wanted to help."
        elif age_group == "7-10":
            s3 = f"Luckily, {character_name} wasn't alone, as {companion} appeared and offered to join the quest."
        else:
            s3 = f"Just when the challenge seemed overwhelming, {companion} emerged, recognizing {character_name}'s determination and offering support."
        
        sentences.append(create_simple_sentence(s3, min_words, max_words))
    
    # Sentence 4: Build excitement
    if num_sentences >= 4:
        if age_group == "3-6":
            s4 = f"{character_name} felt excited and brave."
        elif age_group == "7-10":
            s4 = f"With renewed confidence, {character_name} and the companion prepared to face whatever came next."
        else:
            s4 = f"Together, they understood that the stakes were high, but their combined strength and determination would see them through."
        
        sentences.append(create_simple_sentence(s4, min_words, max_words))
    
    return " ".join(sentences) + " "


def generate_page_4(character_name: str, special_ability: str, age_group: str) -> str:
    """Generate Page 4: Challenge overcome."""
    age_config = AGE_CONFIGS[age_group]
    num_sentences = age_config["page_sentences"][3]
    min_words, max_words = age_config["sentence_length"]
    
    sentences = []
    
    # Sentence 1: Face challenge
    if age_group == "3-6":
        s1 = f"The challenge was hard but {character_name} was brave."
    elif age_group == "7-10":
        s1 = f"When the moment of truth arrived, {character_name} faced the challenge head-on, even though it seemed impossible at first."
    else:
        s1 = f"As the challenge reached its peak, {character_name} confronted the obstacle with a mixture of fear and determination, knowing that this was the moment that mattered most."
    
    sentences.append(create_simple_sentence(s1, min_words, max_words))
    
    # Sentence 2: Use special ability
    if age_group == "3-6":
        s2 = f"{character_name} used the power to {special_ability} and it helped."
    elif age_group == "7-10":
        s2 = f"Remembering the special ability, {character_name} decided to {special_ability}, and this made all the difference."
    else:
        s2 = f"In that critical moment, {character_name} realized that the ability to {special_ability} was exactly what was needed, and with focus and determination, {character_name} used it to overcome the obstacle."
    
    sentences.append(create_simple_sentence(s2, min_words, max_words))
    
    # Sentence 3: Demonstrate growth
    if age_group == "3-6":
        s3 = f"{character_name} learned to be clever and strong."
    elif age_group == "7-10":
        s3 = f"Through this experience, {character_name} discovered inner strength and cleverness that {character_name} didn't know {character_name} had."
    else:
        s3 = f"This experience revealed to {character_name} that growth comes not from avoiding challenges, but from facing them with courage and using one's unique gifts wisely."
    
    sentences.append(create_simple_sentence(s3, min_words, max_words))
    
    # Sentence 4: Companion helps
    if num_sentences >= 4:
        if age_group == "3-6":
            s4 = f"Together they solved the problem."
        elif age_group == "7-10":
            s4 = f"With the help of the companion and {character_name}'s special ability, they worked together and succeeded."
        else:
            s4 = f"The combination of {character_name}'s unique ability and the companion's support created a powerful synergy that led to success."
        
        sentences.append(create_simple_sentence(s4, min_words, max_words))
    
    return " ".join(sentences) + " "


def generate_page_5(character_name: str, special_ability: str, adventure_type: str, age_group: str) -> str:
    """Generate Page 5: Resolution and growth."""
    age_config = AGE_CONFIGS[age_group]
    num_sentences = age_config["page_sentences"][4]
    min_words, max_words = age_config["sentence_length"]
    
    sentences = []
    
    # Sentence 1: Resolution
    if age_group == "3-6":
        s1 = f"{character_name} completed the adventure successfully."
    elif age_group == "7-10":
        s1 = f"The adventure came to a wonderful conclusion, and {character_name} felt proud of what had been accomplished."
    else:
        s1 = f"As the adventure reached its resolution, {character_name} reflected on the journey and felt a deep sense of accomplishment and fulfillment."
    
    sentences.append(create_simple_sentence(s1, min_words, max_words))
    
    # Sentence 2: Personal growth
    if age_group == "3-6":
        s2 = f"{character_name} learned that using special abilities helps others."
    elif age_group == "7-10":
        s2 = f"{character_name} realized that the ability to {special_ability} was not just a power, but a gift to be shared with others."
    else:
        s2 = f"{character_name} understood that the true value of the ability to {special_ability} lay not in its uniqueness, but in how it could be used to help others and make the world a better place."
    
    sentences.append(create_simple_sentence(s2, min_words, max_words))
    
    # Sentence 3: Positive message and ending
    if num_sentences >= 3:
        theme = adventure_type.replace("_", " ").title()
        if age_group == "3-6":
            s3 = f"{character_name} knew that being brave and kind makes everything better."
        elif age_group == "7-10":
            s3 = f"The message was clear: courage, kindness, and using your gifts wisely can overcome any challenge and bring joy to everyone."
        else:
            s3 = f"{character_name} carried forward the profound lesson that true growth comes from embracing challenges, using one's unique abilities for good, and understanding that every adventure teaches us something valuable about ourselves and the world."
        
        sentences.append(create_simple_sentence(s3, min_words, max_words))
    
    return " ".join(sentences) + " "


def _expand_story(pages: List[str], age_group: str, words_needed: int) -> List[str]:
    """Expand story to meet minimum word count."""
    expanded = []
    for page in pages:
        sentences = re.split(r'[.!?]+', page)
        new_sentences = []
        for sent in sentences:
            if sent.strip():
                words = sent.split()
                if len(words) < AGE_CONFIGS[age_group]["sentence_length"][1]:
                    additions = ["very", "so", "really", "quite", "truly"]
                    if len(words) > 0 and words_needed > 0:
                        words.insert(-1, random.choice(additions))
                        words_needed -= 1
                new_sentences.append(" ".join(words))
        expanded.append(". ".join([s for s in new_sentences if s.strip()]) + ". ")
    return expanded


def _trim_story(pages: List[str], age_group: str, words_to_remove: int) -> List[str]:
    """Trim story to meet maximum word count."""
    trimmed = []
    for page in pages:
        sentences = re.split(r'[.!?]+', page)
        new_sentences = []
        for sent in sentences:
            if sent.strip() and words_to_remove > 0:
                words = sent.split()
                if len(words) > AGE_CONFIGS[age_group]["sentence_length"][0]:
                    fillers = ["very", "so", "really", "quite", "truly"]
                    for filler in fillers:
                        if filler in words and words_to_remove > 0:
                            words.remove(filler)
                            words_to_remove -= 1
                            break
                new_sentences.append(" ".join(words))
            elif sent.strip():
                new_sentences.append(sent.strip())
        trimmed.append(". ".join([s for s in new_sentences if s.strip()]) + ". ")
    return trimmed


def generate_story(
    character_name: str,
    character_type: str,
    special_ability: str,
    age_group: str,
    story_world: str,
    adventure_type: str,
    occasion_theme: Optional[str] = None,
    use_api: bool = False,
    api_key: Optional[str] = None,
    story_text_prompt: Optional[str] = None
) -> Dict[str, any]:
    """
    Generate a complete 5-page story for children.
    
    Args:
        character_name: Name of the main character (e.g., "Luna")
        character_type: Type of character (e.g., "a brave dragon", "a curious fox")
        special_ability: Character's special ability (e.g., "fly through clouds", "talk to animals")
        age_group: Target age group - must be "3-6", "7-10", or "11-12"
        story_world: The world where the story takes place (e.g., "the Enchanted Forest")
        adventure_type: Type of adventure (e.g., "treasure hunt", "rescue mission")
        occasion_theme: Optional theme for special occasions (e.g., "birthday", "holiday")
        use_api: If True, use OpenAI API (requires openai package and api_key)
        api_key: OpenAI API key (required if use_api is True)
    
    Returns:
        Dictionary with the following keys:
        - 'pages': List of 5 strings, each containing one page of the story
        - 'full_story': Complete story as a single string
        - 'word_count': Total word count across all pages
        - 'page_word_counts': List of word counts for each page
    
    Raises:
        ValueError: If age_group is not one of the supported values
        
    Example:
        >>> result = generate_story(
        ...     character_name="Luna",
        ...     character_type="a brave dragon",
        ...     special_ability="fly through clouds",
        ...     age_group="7-10",
        ...     story_world="the Enchanted Forest",
        ...     adventure_type="treasure hunt"
        ... )
        >>> for i, page in enumerate(result['pages'], 1):
        ...     print(f"PAGE {i}: {page}")
    """
    if age_group not in AGE_CONFIGS:
        raise ValueError(f"Invalid age group: {age_group}. Must be one of: 3-6, 7-10, 11-12")
    
    # Use API if requested
    if use_api and api_key:
        try:
            return _generate_with_api(
                character_name, character_type, special_ability, age_group,
                story_world, adventure_type, occasion_theme, api_key, story_text_prompt
            )
        except Exception as e:
            print(f"API error: {e}")
            print("Falling back to template-based generation...")
    
    # Generate pages
    pages = []
    pages.append(generate_page_1(character_name, character_type, special_ability, age_group))
    pages.append(generate_page_2(character_name, story_world, age_group))
    pages.append(generate_page_3(character_name, adventure_type, age_group))
    pages.append(generate_page_4(character_name, special_ability, age_group))
    pages.append(generate_page_5(character_name, special_ability, adventure_type, age_group))
    
    # Verify word count
    full_story = "".join(pages)
    total_words = count_words(full_story)
    age_config = AGE_CONFIGS[age_group]
    min_words, max_words = age_config["total_words"]
    
    # Adjust if needed
    if total_words < min_words:
        pages = _expand_story(pages, age_group, min_words - total_words)
    elif total_words > max_words:
        pages = _trim_story(pages, age_group, total_words - max_words)
    
    full_story = "".join(pages)
    page_word_counts = [count_words(page) for page in pages]
    
    return {
        "pages": pages,
        "full_story": full_story,
        "word_count": count_words(full_story),
        "page_word_counts": page_word_counts
    }


def _generate_with_api(
    character_name: str,
    character_type: str,
    special_ability: str,
    age_group: str,
    story_world: str,
    adventure_type: str,
    occasion_theme: Optional[str],
    api_key: str,
    story_text_prompt: Optional[str] = None
) -> Dict[str, any]:
    """Generate story using OpenAI API. If story_text_prompt is provided, use it; otherwise generate prompt from parameters."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package not installed. Install it with: pip install openai")
    
    client = OpenAI(api_key=api_key)
    
    # Use provided prompt if available, otherwise generate one (for backward compatibility)
    if story_text_prompt:
        prompt = story_text_prompt
    else:
        # Fallback: generate prompt from parameters (for backward compatibility)
        age_config = AGE_CONFIGS[age_group]
        environment_details = get_environment_details(story_world)
        
        prompt = f"""Create a personalized 5-page children's storybook.

CHARACTER INFORMATION:
- Name: {character_name}
- Type: {character_type}
- Special Ability: {special_ability}
- Age Group: {age_group}

STORY CONFIGURATION:
- World: {story_world}
- Environment Details: {environment_details}
- Adventure Type: {adventure_type}
- Occasion Theme: {occasion_theme if occasion_theme else 'None'}

AGE-APPROPRIATE REQUIREMENTS FOR {age_group}:
- Sentence Length: {age_config['sentence_length'][0]}-{age_config['sentence_length'][1]} words per sentence
- Total Word Count: {age_config['total_words'][0]}-{age_config['total_words'][1]} words across all 5 pages
- Sentence Structure: {age_config['sentence_structure']}

STORY STRUCTURE (MANDATORY):

PAGE 1 ({age_config['page_sentences'][0]} sentences):
- Introduce {character_name}
- Establish {character_type} identity
- Reveal {special_ability}
- Set positive, welcoming tone

PAGE 2 ({age_config['page_sentences'][1]} sentences):
- {character_name} discovers portal/entrance to {story_world}
- Describe first impressions of the world
- Establish what draws them into the adventure

PAGE 3 ({age_config['page_sentences'][2]} sentences):
- Adventure begins: {adventure_type}
- Introduce challenge or quest objective
- Optional: Introduce companion character
- Build excitement and stakes

PAGE 4 ({age_config['page_sentences'][3]} sentences):
- {character_name} faces main challenge
- Uses {special_ability} to overcome obstacle
- Demonstrates growth or cleverness
- Companion helps if present

PAGE 5 ({age_config['page_sentences'][4]} sentences):
- Resolution of adventure
- {character_name}'s personal growth
- Positive message about {adventure_type}
- Warm, satisfying ending

CRITICAL REQUIREMENTS:
1. Character name must appear at least once per page
2. Special ability must be referenced in pages 1, 4, and 5
3. All sentences must be age-appropriate for {age_group}
4. Maintain consistent tone throughout
5. NO scary, violent, or inappropriate content
6. Positive, empowering message
7. Educational value appropriate to age
8. Gender-neutral language unless character gender specified

Format the output as:
PAGE 1:
[content]

PAGE 2:
[content]

PAGE 3:
[content]

PAGE 4:
[content]

PAGE 5:
[content]
"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a children's story writer who creates age-appropriate, positive, and educational stories."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1500
    )
    
    story_text = response.choices[0].message.content.strip()
    
    # Parse the response into pages
    pages = []
    for i in range(1, 6):
        page_match = re.search(rf'PAGE {i}:\s*(.*?)(?=PAGE {i+1}:|$)', story_text, re.DOTALL)
        if page_match:
            pages.append(page_match.group(1).strip() + " ")
        else:
            # Fallback: split by paragraphs
            paragraphs = story_text.split('\n\n')
            if len(paragraphs) >= i:
                pages.append(paragraphs[i-1].strip() + " ")
            else:
                pages.append("")
    
    full_story = "".join(pages)
    page_word_counts = [count_words(page) for page in pages]
    
    return {
        "pages": pages,
        "full_story": full_story,
        "word_count": count_words(full_story),
        "page_word_counts": page_word_counts
    }

