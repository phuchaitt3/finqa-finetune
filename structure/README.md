### Your New, Efficient Workflow

Now, you can work from your command line terminal.

**Step 1: Train the model (Run this only once, or whenever you want to retrain)**
```bash
python train.py
```
This will take time. It will create the `finqa_t5_final_model` directory containing your trained model.

**Step 2: Generate predictions (Run this anytime you want to test)**
```bash
python predict.py
```
This will be much faster. It loads your saved model and creates `predictions_final.json`.

**Step 3: Evaluate the results**
```bash
python ./code/evaluate/evaluate.py predictions_final.json ./dataset/test.json
```
This runs the official evaluation and gives you the accuracy scores.