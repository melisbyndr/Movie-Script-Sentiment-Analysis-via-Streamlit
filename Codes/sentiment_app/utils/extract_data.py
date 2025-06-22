import re
import pandas as pd

def split_into_scenes(text):
    """
    Splits the script text into scenes. It starts collecting scenes only after the
    first major scene heading (INT. or EXT.) is found, ignoring the title page.
    """
    lines = text.splitlines()
    scenes = []
    current_scene_content = []
    current_scene_heading = "Initial Content (Pre-Script)"
    script_started = False

    scene_heading_regex = re.compile(r'^(INT\.?\/EXT\.?|EXT\.?|INT\.?|EST\.?)')

    for line in lines:
        line_stripped = line.strip()

        if not line_stripped:
            continue
            
        # A line is a scene heading if it starts with common screenplay abbreviations
        if scene_heading_regex.match(line_stripped):
            if script_started and current_scene_content:
                # Save the previous scene
                scenes.append({
                    "scene_heading": current_scene_heading,
                    "content": "\n".join(current_scene_content)
                })
            
            # Start a new scene
            script_started = True
            current_scene_heading = line_stripped
            current_scene_content = []
        elif script_started:
            # Add line to the current scene if the script has started
            # Skip common transition cues
            if not line_stripped.isupper() or not line_stripped.endswith('TO:'):
                current_scene_content.append(line)

    # Add the last scene
    if script_started and current_scene_content:
        scenes.append({
            "scene_heading": current_scene_heading,
            "content": "\n".join(current_scene_content)
        })

    return pd.DataFrame(scenes)


def extract_clean_dialogues(scene_id, scene_text):
    """
    Extracts characters and their dialogues from a scene's text using indentation heuristics.
    """
    lines = scene_text.splitlines()
    results = []
    current_character = None
    current_lines = []

    # Heuristic indentation thresholds
    CHAR_INDENT_MIN = 25
    DIALOGUE_INDENT_MIN = 15

    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        if not stripped:
            continue

        # Character Detection: Indented, all-caps, and not a parenthetical.
        is_character = (
            indent >= CHAR_INDENT_MIN and
            stripped.isupper() and
            '(' not in stripped and
            ')' not in stripped
        )

        # Dialogue Detection: Follows a character and is indented.
        is_dialogue = current_character and indent >= DIALOGUE_INDENT_MIN

        if is_character:
            # If we find a new character, save the previous one's dialogue first.
            if current_character and current_lines:
                results.append({
                    "scene_id": scene_id,
                    "character": current_character,
                    "clean_dialogue": " ".join(current_lines)
                })
            
            # Set the new current character and reset dialogue lines.
            current_character = stripped
            current_lines = []
        
        elif is_dialogue:
            # Append dialogue line, skipping parentheticals.
            if not stripped.startswith('(') and not stripped.endswith(')'):
                current_lines.append(stripped)
        
        else:
            # This line is likely an action line (not indented enough).
            # Save any pending dialogue and reset.
            if current_character and current_lines:
                results.append({
                    "scene_id": scene_id,
                    "character": current_character,
                    "clean_dialogue": " ".join(current_lines)
                })
            current_character = None
            current_lines = []

    # After the loop, save any remaining dialogue.
    if current_character and current_lines:
        results.append({
            "scene_id": scene_id,
            "character": current_character,
            "clean_dialogue": " ".join(current_lines)
        })

    return results


def extract_script_info(text):
    """
    Master function to extract scene-wise cleaned dialogue dataframe from script text.
    """
    scenes_df = split_into_scenes(text)

    all_dialogues = []
    if "content" in scenes_df.columns:
        for idx, row in scenes_df.iterrows():
            scene_id = idx
            scene_text = row["content"]
            scene_dialogues = extract_clean_dialogues(scene_id, scene_text)
            all_dialogues.extend(scene_dialogues)

    if all_dialogues:
        return pd.DataFrame(all_dialogues)
    else:
        # Return an empty dataframe with the expected columns if no dialogues are found
        return pd.DataFrame(columns=['scene_id', 'character', 'clean_dialogue'])
