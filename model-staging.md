In the **new MLflow Model Registry UI**, **"stages"** like `Staging`, `Production`, and `Archived` are **deprecated**.  
Instead, MLflow now recommends **using "custom tags"** to manage stage-like behavior.

---

**What this means for you**:  
✅ You can **still simulate stages** (like assigning a model to "Production" or "Staging"),  
but you now do it **through tags**, not through the old "Stage" system.

---

### How to assign a "stage" now in the new MLflow UI:

1. **Go to your registered model** in the MLflow UI.
2. **Select a model version** you want to "assign a stage" to.
3. **Click on "Edit tags"** (you'll find this option in the model version page).
4. **Add a new tag** like:
   - **Key:** `stage`
   - **Value:** `Production`, `Staging`, `Archived`, etc.
   
   (You can choose whatever naming convention you want — it's flexible.)

---

### Example:

| Tag Key | Tag Value  |
|:-------|:-----------|
| stage  | Production |
| stage  | Staging    |
| stage  | Archived   |

This way, you manually "assign a stage" using tags.

---

### If you prefer using **code (Python)** via `mlflow`:

```python
import mlflow

# Set the tag for a particular model version
client = mlflow.tracking.MlflowClient()
client.set_model_version_tag(
    name="your-registered-model-name",
    version=1,  # specify the version you want
    key="stage",
    value="Production"  # or "Staging" or "Archived"
)
```

---

### Why did MLflow deprecate stages?
- To make it more flexible and **customizable**.
- Some companies wanted **more than 3 stages** (for example: "QA", "Pre-Production", "Testing", etc).
- Tags allow you to define **your own lifecycle** without being locked into the old model.