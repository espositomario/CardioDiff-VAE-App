Open the [Streamlit App](https://cardiodiff-vae.streamlit.app/)


### Run locally 


1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

To prevent streamlit app go to sleep:




1. Run crontab -e from the terminal
2. Add this:0 12,16 * * * curl https://cardiodiff-vae.streamlit.app/
