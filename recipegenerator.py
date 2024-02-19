# Import recipe generation function and other required libraries 
from cgitb import text
import tkinter as tk
from tkinter import ttk
from transformers import FlaxAutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import pandas as pd 
import random
import nltk
from nltk.translate.meteor_score import meteor_score


MODEL_NAME_OR_PATH = "flax-community/t5-recipe-generation"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
model = FlaxAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)

prefix = "items: "
# generation_kwargs = {
#     "max_length": 512,
#     "min_length": 64,
#     "no_repeat_ngram_size": 3,
#     "early_stopping": True,
#     "num_beams": 5,
#     "length_penalty": 1.5,
# }
generation_kwargs = {
    "max_length": 512,
    "min_length": 64,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "top_k": 60,
    "top_p": 0.95
}


special_tokens = tokenizer.all_special_tokens
tokens_map = {
    "<sep>": "--",
    "<section>": "\n"
}
def skip_special_tokens(text, special_tokens):
    for token in special_tokens:
        text = text.replace(token, "")

    return text.strip()


def target_postprocessing(texts, special_tokens):
    if not isinstance(texts, list):
        texts = [texts]

    new_texts = []
    for text in texts:
        text = skip_special_tokens(text, special_tokens)

        for k, v in tokens_map.items():
            text = text.replace(k, v)

        new_texts.append(text)

    return new_texts

def generation_function(texts):
    _inputs = texts if isinstance(texts, list) else [texts]
    inputs = [prefix + inp for inp in _inputs]
    inputs = tokenizer(
        inputs,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="jax"
    )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_kwargs
    )
    generated = output_ids.sequences
    generated_recipe = target_postprocessing(
        tokenizer.batch_decode(generated, skip_special_tokens=False),
        special_tokens
    )
    return generated_recipe

def meteor_evaluation(ground_truth_recipes, generated_recipes):         #meteor evalutation function
    meteor_scores = []

    for gt_recipe, gen_recipe in zip(ground_truth_recipes, generated_recipes):
        # Tokenize each recipe into lists of strings (steps)
        gt_tokens = gt_recipe.split()
        gen_tokens = gen_recipe.split()

        meteor_score_value = meteor_score([gt_tokens], gen_tokens)
        meteor_scores.append(meteor_score_value)

    return sum(meteor_scores) / len(meteor_scores)

def find_matching_instructions(user_input):
    # Provide the path to your dataset CSV file and the column name for ingredients and instructions
    dataset_csv_file_path = "combined_recipes.csv"  
    ingredient_column_name = "ingredients"  
    instructions_column_name = "instructions"  

    # Read the dataset CSV file
    df = pd.read_csv(dataset_csv_file_path)

    # Convert user_input and ingredients in the DataFrame to lowercase for case-insensitive matching
    user_input = [item.lower().strip() for item in user_input]
    df[ingredient_column_name] = df[ingredient_column_name].str.lower().str.strip()

    # Initialize a list to store matching instructions
    matching_instructions = []

    for item in user_input:
        # Find rows where the item is present in the ingredients column
        matches = df[df[ingredient_column_name].str.contains(item)]

        # Append the matching instructions to the list
        matching_instructions.extend(matches[instructions_column_name].tolist())

    # If there are more than 10 matching instructions, randomly select 10
    # if len(matching_instructions) > 10:
    #     matching_instructions = random.sample(matching_instructions, 10)
    
    #print(matching_instructions)

    return matching_instructions

# Provide the path to your dataset CSV file and the column name for ingredients
dataset_csv_file_path = "combined_recipes.csv"  # Replace with the actual path to your CSV file
ingredient_column_name = "ingredients"  # Replace with the actual column name in your CSV that contains the ingredients

# Read the dataset CSV file and extract valid items
df = pd.read_csv(dataset_csv_file_path)
dataset_items = df[ingredient_column_name].str.lower().tolist()


def is_input_in_dataset(input_items, dataset_items):
    input_items = [item.strip().lower() for item in input_items]
    dataset_items = [item.strip().lower() for item in dataset_items] 

    for input_item in input_items:
        found = False
        for dataset_item in dataset_items:
            if input_item in dataset_item:
                found = True
                break
        if not found:
            return False

    return True



def on_ok_click():
    # Initialize the items list to an empty list
    items = []
    itemscheck = []
    headline=""
    outputlist = []

    # Get the input text from the Text widget and split it into a list
    input_paragraph = input_text.get("1.0", tk.END) #get input
    individualingredients = input_paragraph.split(",")

     # Clean up the individual ingredients and add them to the 'items' list
    itemscheck.extend([ingredient.strip() for ingredient in individualingredients])


    if not is_input_in_dataset(itemscheck, dataset_items):
        output_text.config(state='normal')
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, "Apologies, one or more of your ingredient(s) is invalid") #give friendly message if theres an invalid input from user
        output_text.config(state='disabled')
        return

    
    else:
        # Run the generation_function with the updated 'items' list
        directionslist = []
        items.append(input_paragraph) #append into items list
        generated = generation_function(items)
        matchinginstructions = find_matching_instructions(input_paragraph) #call the matching instruction function for ground truth used in meteor testing
        for text in generated:
            sections = text.split("\n")
            for section in sections:
                section = section.strip()
                if section.startswith("title:"):    #replace the headers from the generated text (applys to title, ingredients and directions)
                    section = section.replace("title:", "")
                    headline = "RECIPE"
                elif section.startswith("ingredients:"):
                    section = section.replace("ingredients:", "")
                    headline = "INGREDIENTS"
                elif section.startswith("directions:"):
                    section = section.replace("directions:", "")
                    headline = "DIRECTIONS"
                    directionslist.append(section)
                    print(directionslist)
                    #newdirectionlist = "".join(directionslist)
                    
                
                #this is to append all replacements and the information together
                if headline == "RECIPE":
                    title = f"[{headline}]: {section.strip().capitalize()}\n"
                    topline = "-" * 30 
                    bottomline = "-" * 30
                    #outputlist.append(topline)
                    outputlist.append(title)
                    #outputlist.append(bottomline)

                else:
                    section_info = [f"  - {i+1}: {info.strip().capitalize()}" for i, info in enumerate(section.split("--"))] #splits each instruction / ingredient by the "--" seperator in the generated.
                    recipe = f"[{headline}]:\n" + "\n".join(section_info) +"\n" #joining the headline and the info
                    outputlist.append(recipe) #appending each item, such as title, ingredients, direction and its respective info into a list
                    

        # Format and display the generated recipe
        outputrecipe = "\n".join(outputlist) # concatenate each section with a line separator
        output_text.config(state='normal')  # Set the state to normal to enable editing temporarily
        output_text.delete("1.0", tk.END)  # Clear previous content
        output_text.insert(tk.END, outputrecipe) #insert fully joined recipe 
        output_text.config(state='disabled')  # Set the state back to disabled to make it read-only again
        meteor_score_avg = meteor_evaluation(matchinginstructions,directionslist)
        #print(outputlist[2])
        #print(f"METEOR Score: {meteor_score_avg:.4f}")
        meteorscore = f"METEOR Score: {meteor_score_avg:.4f}"
        score_label.config(text=meteorscore)


def on_exit_click():
    root.quit()

# Create the GUI
root = tk.Tk()
root.title('Recipe Generator')
root.configure(bg='white')

# Create a rounded style for the buttons
s = ttk.Style()
s.configure('Rounded.TButton', borderwidth=0, focuscolor='white', focusthickness=0, font=('Helvetica', 12))

main_frame = ttk.Frame(root, padding=10)
main_frame.pack(expand=True, fill='both')

input_label = ttk.Label(main_frame, text='Enter your Available Ingredients')
input_label.pack(pady=5)

# Use a Text widget for the multi-line input
input_text = tk.Text(main_frame, wrap=tk.WORD, width=30, height=5)
input_text.pack(pady=5)

output_label = ttk.Label(main_frame, text='Generated Recipe Goes Below:')
output_label.pack(pady=2)

output_text = tk.Text(main_frame, wrap=tk.WORD, width=80, height=20, state="disabled")
output_text.pack(pady=5)

score_label = ttk.Label(main_frame, text='METEOR SCORE: ')
score_label.pack(pady=2)

# Centering the buttons in a new frame
button_frame = ttk.Frame(main_frame)
button_frame.pack(pady=10)

ok_button = ttk.Button(button_frame, text='Generate', style='Rounded.TButton', command=on_ok_click)
ok_button.pack(side=tk.LEFT, padx=5)

exit_button = ttk.Button(button_frame, text='Exit', style='Rounded.TButton', command=on_exit_click)
exit_button.pack(side=tk.RIGHT, padx=5)

# Set the background color using the style option
s.configure('Rounded.TButton', background='#87CEEB', foreground='dark blue')

root.mainloop()
