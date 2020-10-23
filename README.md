## Notes

The `requirements.txt` file should list all Python libraries that this project
depends on, and they will be installed using:

```
pip install -r requirements.txt
```

The `config.py` lists all the local paths to the necessary data. If this code is 
to run on any other machine, every data source listed on the file should be downloaded
and the paths changed accordingly.

Since the above condition might not be possible. I have included `.html` versions of 
two of the `.md` markdowns, should the evaluator want to see cell outputs.



# Please read

### Working with notebooks
If you’re working with notebooks, please make sure to also include a `.md` version since GitHub doesn’t always properly render the content of the notebook. (The [jupytext](https://github.com/mwouts/jupytext) library can help to change the format)

### How to submit your results
Please send us a csv file with your predictions for the evaluation test by 2020-10-24 12pm CET by email at elena.pedrini@bain.com with the subject `[data challenge submission]`

The submission file shall include the following columns `[tile_h, tile_v, prediction]`. You can use submission_sample.csv as a template (link on slide 3)​

The name of your submission file shall conform to this pattern <yyyymmddHHMMSS>_<github-username>.csv​

- 20201020131210_johnsnow.csv is a valid filename​

- 20201022-johnsnow.csv is not a valid filename​

Feel free to send your intermediary predictions beforehand, just to make sure that we can score the file.
