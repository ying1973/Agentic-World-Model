import re
import json
from typing import List, Optional, Tuple, Any, Dict
from PIL import Image
from io import BytesIO
import base64
import random
import tempfile
import os
import cv2
import sys

from awm.media_cache import global_media_cache
from awm.tools import Tool, get_tools, synthesis_chatgpt


class MultiToolReActAgent:
    """Enhanced ReAct Agent with Multi-Tool Composition and Self-Evaluation"""
    def __init__(self, tools: List[Tool], max_iterations: int = 5, verbose: bool = True):
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.verbose = verbose


    def _create_prompt_template(self) -> str:
        """Create ReAct prompt template with multi-tool composition support"""
        # def _escape_braces(s: str) -> str:
        #     return s.replace('{', '{{').replace('}', '}}')

        # items = [
        #     f"{name}: {_escape_braces(tool.description) if isinstance(tool.description, str) else tool.description}"
        #     for name, tool in self.tools.items()
        # ]
        # sample_size = max(1, int(len(items) * 0.7))
        # sampled_tool_descriptions = random.sample(items, sample_size)
        # tool_descriptions = "\n".join(sampled_tool_descriptions)
        tools_payload = []

        for name, tool in self.tools.items():
            if isinstance(tool.description, str):
                try:
                    desc_obj = json.loads(tool.description)
                except json.JSONDecodeError:
                    desc_obj = {"description": tool.description}
            else:
                desc_obj = tool.description

            tools_payload.append({
                "name": name,
                **desc_obj
            })
        sample_size = max(1, int(len(tools_payload) * 1.0))
        sampled_tools = random.sample(tools_payload, sample_size)
        sampled_tool_descriptions = json.dumps(
            sampled_tools,
            ensure_ascii=False,
            indent=2
        )

        template = f"""
You are an intelligent assistant that solves problems through **creative multi-tool exploration**.
Your goal: discover diverse tool combinations and validate results through multiple approaches.

========================================
AVAILABLE TOOLS
========================================
{sampled_tool_descriptions}

========================================
CORE PRINCIPLES
========================================

1. **Novelty First**: Each iteration should try NEW tools or NEW combinations.
   - Explicitly check: "Have I used this exact tool/combination before?"
   - If yes, explain why repeating is necessary, or choose differently.

2. **Combination Creativity**: 
   - You can use a text generation tool to create a descriptive prompt, then pass it to a general-purpose video generation tool along with the original image to generate the next-moment video.
   - You can use a text generation tool to create a descriptive prompt, then pass it to an image editing tool along with the original image to generate the next-state image.
   - You can compare the differences between the execution results of two tools with similar functions and select the optimal one.
   - You can use an image editing tool to produce an updated first-frame image, then input it into a general-purpose video generation tool to create a more realistic video.
   - Remember that you can always use the output of one tool as the input for another.
   - There are many other multi-tool combination methods here—please actively explore them.
   - Please rely on tools to complete any operations as much as possible and use your own linguistic abilities as little as possible.

3. **Mandatory Diversity**: 
   - For tasks solvable by multiple tools: try at least 2-3 different tools
   - For complex tasks: use at least 3-4 different tool compositions
   - Keep a mental "tool usage count" - favor underused tools

========================================
STRICT EXECUTION CYCLE
========================================

Step 1: <think>
Required elements:
- **Task understanding**: What exactly does the user need?
- **History review**: What have I already tried? Are they helpful or not? Explain WHY
- **Combination ideas**: Based on the task and history, describe 2-3 potential multi-tool strategies with WHY each might help
- **Next action**: Choose ONE action in One strategy that use previous information and adds NEW information
- **Diversity reasoning**: Explicitly state how this action differs from previous attempts
</think>

Step 2: <action>
{{
    "tool": "exact_tool_name",
    "input": "specific_input_here"
}}
</action>

CRITICAL: After </action>, you MUST STOP GENERATION IMMEDIATELY.
   - DO NOT generate <observation> yourself
   - DO NOT continue with any text
   - Wait for the system to provide the real observation

Step 3: [System provides] <observation>
real_tool_output_here
</observation>

Step 4: <evaluate>
Required elements:
- **Result quality**: Does this output answer part/all of the task? How well?
- **New information**: What NEW insight did this provide?
- **Gaps remaining**: What's still missing or uncertain?
- **Confidence**: How confident am I in this result? (low/medium/high)
- **Next direction**: Should I validate, refine, try alternatives, or conclude?
</evaluate>

Step 5: Return to Step 1 (<think>) OR output <final_answer>

========================================
ANTI-REPETITION RULES
========================================

Before each <action>, verify in <think>:
"I have NOT used this exact tool with similar input before" OR
"I am repeating because: [specific reason, e.g., refining with new parameters from previous output]"

If you catch yourself repeating without good reason, STOP and choose a different tool.

========================================
TERMINATION CONDITIONS
========================================

You may output <final_answer> ONLY when ALL are true:
1. You've tried at least 2-3 different tools or compositions (unless trivial task)
2. Latest <evaluate> shows "confidence: high" 
3. Your next <think> explicitly states: "No additional tools would meaningfully improve this result because..."
4. You've considered validation/verification and deemed it unnecessary (with reasoning)

Format:
<final_answer>
output the best tool operation result here:
[output text/image_id/video_id]
</final_answer>

========================================
EXPLORATION INCENTIVES
========================================

Bonus credit for:
- Discovering non-obvious tool combinations
- Finding creative ways to chain tool outputs
- Comparing multiple approaches before deciding
- Using tools you haven't used yet in this session
- Validating results through independent methods

Red flags (avoid):
- Using the same tool 2+ times without evolution
- Accepting tool operation result without evaluation
- Ignoring available tools that could add value
- Concluding with <final_answer> after only 1 tool call (unless truly trivial)

========================================
QUESTION
========================================
{{question}}

========================================
EXECUTION HISTORY
========================================
{{history}}

Begin your first <think> block now."""
        
        safe = template.replace('{', '{{').replace('}', '}}')
        safe = safe.replace('{{question}}', '{question}').replace('{{history}}', '{history}')
        return safe, sampled_tool_descriptions


    def _register_media(self, image: Any):
        """注册图片到全局缓存并返回 (image_id, base64)"""
        if isinstance(image, str):
            with open(image, "rb") as f:
                img_bytes = f.read()
        elif isinstance(image, Image.Image):
            buf = BytesIO()
            image.convert("RGB").save(buf, format="PNG")
            img_bytes = buf.getvalue()
        else:
            raise ValueError("Unsupported image type")

        media_id, img_base64 = global_media_cache.register(img_bytes)
        return media_id, img_base64


    def _call_chatgpt(self, prompt: str, image: Optional[Any], history: str) -> str:
        media_ids, video_ids = self._extract_media_and_video_ids_from_history(history)

        image_payload = self._build_media_payload_for_llm(media_ids, video_ids)

        if image is not None:
            media_id, img_base64 = self._register_media(image)
            image_payload.insert(0, {
                "type": "user_input_image",
                "id": media_id,
                "data": img_base64
            })

        response = synthesis_chatgpt(prompt, images=image_payload)
        return response.strip()
    

    def _parse_response(self, text: str) -> Tuple[Optional[str], Optional[Dict], Optional[str], Optional[str]]:
        # Extract think
        think_match = re.search(r'\<think\>(.*?)\</think\>', text, re.DOTALL)
        think = think_match.group(1).strip() if think_match else ""
        
        # Check for final_answer
        final_answer_match = re.search(r'\<final_answer\>(.*?)\</final_answer\>', text, re.DOTALL)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            return think, None, None, final_answer
        
        # Extract action (JSON format)
        action_match = re.search(r'\<action\>(.*?)\</action\>', text, re.DOTALL)
        action_dict = None
        if action_match:
            action_json = action_match.group(1).strip()
            # Remove potential code block markers
            action_json = re.sub(r'```json\s*|\s*```', '', action_json)
            action_dict = json.loads(action_json)

        # Extract evaluate
        evaluate_match = re.search(r'\<evaluate\>(.*?)\</evaluate\>', text, re.DOTALL)
        evaluate = evaluate_match.group(1).strip() if evaluate_match else ""
        
        return think, action_dict, evaluate, None
    

    def _execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        # 工具名检查
        if tool_name not in self.tools:
            return {
                "error": f"Tool '{tool_name}' does not exist.",
                "available_tools": list(self.tools.keys())
            }
        # 输入格式检查
        if not isinstance(tool_input, dict):
            return {
                "error": f"Tool input for '{tool_name}' must be a JSON object (dict)."
            }
        tool_fn = self.tools[tool_name].func
        result = tool_fn(**tool_input)
        return result
    

    def _is_image_file(self, path: str) -> bool:
        return path.lower().endswith((".png", ".jpg", ".jpeg"))


    def _is_video_file(self, path: str) -> bool:
        return path.lower().endswith((".mp4", ".avi", ".mov"))
        

    def _extract_video_keyframes(self, video_path: str, num_frames: int = 5):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []

        # 均匀采样关键帧
        idxs = [int(total * i / (num_frames + 1)) for i in range(1, num_frames + 1)]
        frame_ids = []

        # prepare save directory and base name
        vid_dir = os.path.dirname(video_path) or "."
        vid_base = os.path.splitext(os.path.basename(video_path))[0]

        count = 1
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            try:
                # encode to PNG bytes
                _, buf = cv2.imencode(".png", frame)
                img_bytes = buf.tobytes()

                # save to same directory as video with specified name
                out_name = f"{vid_base}_sampleframe{count}.png"
                out_path = os.path.join(vid_dir, out_name)
                with open(out_path, "wb") as wf:
                    wf.write(img_bytes)

                # register saved image bytes to global media cache
                media_id, _ = global_media_cache.register(img_bytes)
                frame_ids.append(media_id)
                count += 1
            except Exception:
                # fallback: skip this frame
                continue

        cap.release()
        return frame_ids


    def _extract_media_and_video_ids_from_history(self, history: str):
        media_ids = set()
        video_ids = set()

        # [IMAGE_ID]: media_xxx
        for m in re.findall(r'\[IMAGE_ID\]\s*:\s*(media_[a-fA-F0-9]+)', history):
            media_ids.add(m)

        # [VIDEO_ID]: video_xxx
        for v in re.findall(r'\[VIDEO_ID\]\s*:\s*(video_[a-fA-F0-9]+)', history):
            video_ids.add(v)

        return list(media_ids), list(video_ids)


    def _build_media_payload_for_llm(self, media_ids: list, video_ids: list):
        payload = []

        for m_id in media_ids:
            bytes_data = global_media_cache.get_bytes(m_id)
            b64 = base64.b64encode(bytes_data).decode()
            payload.append({
                "type": "image",
                "id": m_id,
                "data": b64
            })

        for video_id in video_ids:
            frame_ids = global_media_cache.get_video_frames(video_id)
            for fid in frame_ids:
                bytes_data = global_media_cache.get_bytes(fid)
                b64 = base64.b64encode(bytes_data).decode()

                payload.append({
                    "type": "video_frame",
                    "video_id": video_id,
                    "id": fid,
                    "data": b64
                })

        return payload


    def run(self, question: str, image: Optional[Any] = None) -> Tuple[str, Dict]:
        if self.verbose:
            print(f"\n{'-'*100}")
            print(f"❓ Question: {question}")
            print(f"{'-'*100}\n")
        
        history = ""


        # Prepare trajectory recording
        trajectory = {
            "conversations": [],
            "images": None,
            "videos": None,
            "system": None,
            "tools": None,
        }

        # Helper to obtain a local image path for non-path image objects
        def _ensure_local_image_path(img: Any) -> Optional[str]:
            if img is None:
                return None
            if isinstance(img, str) and os.path.exists(img):
                return img
            try:
                if 'Image' in globals() and isinstance(img, Image):
                    fd, tmp_path = tempfile.mkstemp(suffix=".png")
                    os.close(fd)
                    img.convert("RGB").save(tmp_path, format="PNG")
                    return tmp_path
            except Exception:
                pass
            return None

        system_prompt, sampled_tool_descriptions = self._create_prompt_template()
        trajectory["system"] = system_prompt.format(
            question=question, 
            history=history
        )
        trajectory["tools"] = sampled_tool_descriptions

        # Add initial human message, include image placeholder if image provided
        human_value = question
        if image is not None:
            human_value = "<image> " + human_value
            img_local = _ensure_local_image_path(image)
            if img_local:
                if trajectory["images"] is not None:
                    trajectory["images"].append(img_local)
                else:
                    trajectory["images"] = [img_local]
                # trajectory["images"].append(img_local) if trajectory["images"] != None else trajectory["images"] = [img_local]
        trajectory["conversations"].append({"from": "human", "value": human_value})
        

        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"🔄 Iteration {iteration + 1}/{self.max_iterations}")
                print("-" * 100)

            # Build prompt
            prompt = system_prompt.format(
                question=question,
                history=history
            )
            
            # Generate response from ChatGPT
            response = self._call_chatgpt(prompt, image, history)
            
            if self.verbose:
                print(f"🤖 Model Response:\n{response}\n")

            # Record model response as a single 'gpt' entry
            # trajectory["conversations"].append({"from": "gpt", "value": response})
            try:
                replaced_response = response
                found_image_ids = re.findall(r'(media_[a-fA-F0-9]+)', response)
                found_video_ids = re.findall(r'(video_[a-fA-F0-9]+)', response)

                # materialize image ids to local temp files and replace in response
                for mid in list(found_image_ids):
                    local_path = global_media_cache.materialize_to_temp_file(mid)
                    replaced_response = replaced_response.replace(mid, '<image>')
                    if trajectory["images"] is not None:
                        trajectory["images"].append(local_path)
                    else:
                        trajectory["images"] = [local_path]
                    # trajectory["images"].append(local_path) if trajectory["images"] != None else trajectory["images"] = [local_path]

                # get video paths and replace
                for vid in list(found_video_ids):
                    video_path = global_media_cache.get_video_path(vid)
                    replaced_response = replaced_response.replace(vid, '<video>')
                    if trajectory["videos"] is not None: 
                        trajectory["videos"].append(video_path)
                    else:
                        trajectory["videos"] = [video_path]
                    # trajectory["videos"].append(video_path) if trajectory["videos"] != None else trajectory["videos"] = [video_path]

                # finally, add the modified model response into trajectory as a gpt turn
                trajectory["conversations"].append({"from": "gpt", "value": replaced_response})
            except Exception as e:
                # on any failure, fall back to original response
                print(f"Error {e}: Failed to extra image_path/video_path from model's response. Continue with raw response")
                trajectory["conversations"].append({"from": "gpt", "value": response})

            # Parse response
            think, action_dict, evaluate, final_answer = self._parse_response(response)
            if evaluate is not None:
                history += f"<evaluate> {evaluate} </evaluate>\n"

            # Check for final answer
            if final_answer:
                if self.verbose:
                    print(f"\n✅ <final_answer>: {final_answer}")
                    print(f"{'='*100}\n")
                return final_answer, trajectory
            
            # Execute action
            if action_dict and 'tool' in action_dict and 'input' in action_dict:
                print()
                tool_name = action_dict['tool']
                tool_input = action_dict['input']

                # Execute tool
                observation = self._execute_tool(tool_name, tool_input)

                obs_record = None

                if isinstance(observation, str) and os.path.exists(observation):
                    if self._is_image_file(observation):
                        media_id, _ = self._register_media(observation)
                        obs_record = f"[IMAGE_ID]: {media_id}"
                    elif self._is_video_file(observation):
                        frame_ids = self._extract_video_keyframes(observation)
                        video_id = global_media_cache.register_video(observation)
                        global_media_cache.add_video_frames(video_id, frame_ids)
                        obs_record = f"[VIDEO_ID]: {video_id}"
                    else:
                        obs_record = observation
                else:
                    obs_record = str(observation)

                if self.verbose:
                    print(f"👁️  <observation>: {obs_record}")
                
                # Update history
                history += f"\n--- Iteration {iteration + 1} ---\n"
                history += f"<think> {think} </think>\n"
                history += f"<action> {json.dumps(action_dict)} </action>\n"
                history += f"<observation> {obs_record} </observation>\n"

                # Record observation into trajectory

                obs_value = None
                if isinstance(observation, str) and os.path.exists(observation):
                    if self._is_image_file(observation):
                        obs_value = "<image>"
                        if trajectory["images"] is not None:
                            trajectory["images"].append(observation)
                        else:
                            trajectory["images"] = [observation]
                        # trajectory["images"].append(observation) if trajectory["images"] != None else trajectory["images"] = [observation]
                    elif self._is_video_file(observation):
                        obs_value = "<video>"
                        if trajectory["videos"] is not None:
                            trajectory["videos"].append(observation)
                        else:
                            trajectory["videos"] = [observation]
                        # trajectory["videos"].append(observation) if trajectory["videos"] != None else trajectory["videos"] = [observation]
                    else:
                        obs_value = observation
                else:
                    obs_value = str(observation)
                trajectory["conversations"].append({"from": "observation", "value": obs_value})

                # Now the agent needs to evaluate - this will happen in next iteration
                # But we can prompt it to continue with evaluation
                
                # if not evaluate:
                #     # If model didn't provide evaluation, prompt for it
                #     eval_prompt = f"{prompt}\n\n{history}\n\nNow provide your <evaluate> section to assess this result."
                #     eval_response = self._call_chatgpt(eval_prompt, image, history)
                #     _, _, evaluate, _ = self._parse_response(eval_response)
                # if evaluate:
                #     history += f"<evaluate> {evaluate} </evaluate>\n"
                
            else:
                if self.verbose:
                    print("⚠️ Unable to parse valid ACTION, attempting to regenerate...")
                # 新增：提取不到动作时也会在轨迹里添加一个空的observation
                trajectory["conversations"].append({"from": "observation", "value": "None"})
                
                history += f"\n<think> {think} </think>\n"
                history += "[ERROR] Please provide valid <action> with JSON format: {\"tool\": \"tool_name\", \"input\": \"input_value\"}\n"
        
        # Max iterations reached
        if self.verbose:
            trajectory["conversations"].append({"from": "system", "value": "Reached maximum iterations without final answer."})
            print(f"⚠️ Reached maximum iterations ({self.max_iterations})")
            print(f"{'='*70}\n")
        return ("I apologize, but I couldn't complete the task within the allowed iterations. Please try rephrasing your question or breaking it into smaller steps.", trajectory)


# ============= Main Program =============
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run a single AWM ReAct query.")
    parser.add_argument("--question", required=True, help="Task question or instruction.")
    parser.add_argument("--image", default=None, help="Optional input image path.")
    parser.add_argument("--max_iterations", type=int, default=15)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    agent = MultiToolReActAgent(
        tools=get_tools(),
        max_iterations=args.max_iterations,
        verbose=not args.quiet,
    )
    answer, _trajectory = agent.run(args.question, image=args.image)
    print(answer)


if __name__ == "__main__":
    main()
