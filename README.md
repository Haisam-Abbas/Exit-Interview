# Exit-Interview
A Natural Language Processing BART model trained to classify exit interviews into 13 distinct categories.

Hello! I'm Haisam Abbas. The main goal for this project is to create a tool for aiding HR Analytics specifically uncovering the reasons behind employee attrition. 

I believe this tool gives the most value to individuals in consultancy or HR departments in medium to large organizations which deal with copious amounts of unstructured data and legacy systems / processes which have no formal standard. Although it can be helpful to almost anyone.

I used a Zero Shot Facebook/BART model and then finetuned it on a data set of around 2500 rich exit interviews. The data set was created by using a mix of LLM's, research into the leading causes of employee resignations as well as real life exit interview comments.

The model itself is stored on Hugging Face and can be found at : https://huggingface.co/HaisamAbbas1/Exit_Interview . But the GitHub contains the script required for an easy plug and play.

The model process exit comments and classifies them into one of 13 categories:

    "0": "Career change",
    "1": "Entrepreneurship",
    "2": "Further Education",
    "3": "Heavy workload / Burnout",
    "4": "Insufficient training or mentoring",
    "5": "Lack of career growth",
    "6": "Lack of recognition",
    "7": "Limited work-life balance",
    "8": "Management or leadership issues",
    "9": "Other / Unclear",
    "10": "Poor compensation or benefits",
    "11": "Relocation or personal reasons",
    "12": "Toxic culture or team dynamics"


Upon testing the model it gives an labelling accuracy of around 75% - 80%. At its current state it may struggle differentiating between overlapping topics such as Limited work-life balance and Heavy workload / Burnout but scores very high on other stand alone topics such as Relocation, Further Education and Management or leadership issues. Please do validate the end results. ( Maybe in the next iteration I will condense the labels)

This model was my first project relating to Machine Learning and I benefited a lot from opensource materials, code on GitHub and Stack Overflow as well as YouTube tutorials and AI agents.
Being a self taught student of programming I am always looking forward to opportunities to learn. Any feedback is much appreciated!. 
