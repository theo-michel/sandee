import os
from PIL import Image
import base64
from io import BytesIO
from mistralai import Mistral


def analyze_image_in_context(image, client: Mistral, model_name="mistral-small-latest"):
    try:
        # Calculate new dimensions while preserving aspect ratio
        width, height = image.size
        max_dim = 1024
        
        if width > max_dim or height > max_dim:
            if width > height:
                new_width = max_dim
                new_height = int(height * (max_dim / width))
            else:
                new_height = max_dim
                new_width = int(width * (max_dim / height))
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        print(f"Loaded image with size: {image.size}")
        
        # Convert the image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Load few-shot learning example images
        examples = []
        few_shot_dir = "./images_vlm/train"
        example_images = [
            {"path": "image_0_1.jpg", "label": 1},
            {"path": "image_1_0.jpg", "label": 0},
            {"path": "image_2_1.jpg", "label": 1},
            {"path": "image_3_0.jpg", "label": 0}
        ]
        
        for example in example_images:
            try:
                example_path = os.path.join(few_shot_dir, example["path"])
                example_img = Image.open(example_path)
                
                # Resize example images if needed
                width, height = example_img.size
                max_dim = 1024
                if width > max_dim or height > max_dim:
                    if width > height:
                        new_width = max_dim
                        new_height = int(height * (max_dim / width))
                    else:
                        new_height = max_dim
                        new_width = int(width * (max_dim / height))
                    example_img = example_img.resize((new_width, new_height), Image.LANCZOS)
                
                # Convert example image to base64
                example_buffered = BytesIO()
                example_img.save(example_buffered, format="JPEG")
                example_img_str = base64.b64encode(example_buffered.getvalue()).decode()
                
                examples.append({
                    "image": f"data:image/jpeg;base64,{example_img_str}",
                    "label": "yes" if example["label"] == 1 else "no"
                })
            except Exception as e:
                print(f"Error loading example image {example['path']}: {str(e)}")

        # Refined System Prompt: Focus on the specific task and output format
        system_prompt = (
            "Your task is to determine if a specific object, a grey/blue metallic RedBull can, "
            "is the closest object to the camera in the provided image. "
            "Ignore natural objects like seaweed, shells, crabs, or rocks when determining closeness. "
            "Respond ONLY with 'yes' or 'no' in lowercase without punctuation."
        )
        
        # Refined User Prompt: Structure examples clearly and ask the specific question
        user_content = [
            {"type": "text", "text": "Analyze the following examples to understand the task, then evaluate the final image."}
        ]
        
        # Add structured examples to user content
        for idx, example in enumerate(examples):
            # Provide reasoning specific to the task
            reasoning = "The RedBull can (grey/blue metallic object) is visible and closest." if example['label'] == 'yes' else "The RedBull can is either not visible or not the closest object."
            user_content.extend([
                {"type": "text", "text": f"--- Example {idx+1} ---"},
                {"type": "image_url", "image_url": {"url": example["image"]}},
                {"type": "text", "text": f"Reasoning: {reasoning}"},
                {"type": "text", "text": f"Answer: {example['label']}"}
            ])
        
        # Add the final query image and question
        user_content.extend([
             {"type": "text", "text": "--- End of Examples ---"},
             {"type": "text", "text": "Now, analyze this new image:"},
             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}},
             {"type": "text", "text": "Is the RedBull can the closest object to the camera in this image?"}
        ])
        
        # Make API call with refined prompts
        chat_response = client.chat.complete(
            model=model_name,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,  # Adjust temperature for creativity
        )
        
        # Extract response
        response = chat_response.choices[0].message.content
        return response
        
    except Exception as e:
        return f"Error processing image: {str(e)}"