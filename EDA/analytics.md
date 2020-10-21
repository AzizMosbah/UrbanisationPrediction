---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: bainhack
    language: python
    name: bainhack
---

# Analytics with target
    - How do the raw and synthetic features relate to the target ?
    

```python
import os
os.chdir('../')
```

```python
from Pipelines.transformations import pixel_for_NN, table_for_model
```

```python
from datetime import datetime
startTime = datetime.now()
pixel = pixel_for_NN()
print(datetime.now() - startTime)
```

```python
from datetime import datetime
startTime = datetime.now()
table = table_for_model()
print(datetime.now() - startTime)
```

```python
pixel.head()
```

```python
table.head()
```
