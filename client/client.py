import socket
import json
import os
from audio_handler import AudioHandler
import time
from openai import OpenAI
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats as scipy_stats
import plotext as plt
from datetime import datetime

class SkillProgressClient:
    def __init__(self, host='localhost', port=65432, user_name='stranger'):
        self.host = host
        self.port = port
        self.user_name = user_name
        self.audio = AudioHandler()
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))
        print(f"I've got brains")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def send_data(self, data):
        message = json.dumps(data).encode()
        print(f"Communicating with RxInfer")
        try:
            self.client_socket.sendall(message)
            response = self.client_socket.recv(4096)
            
            logging.debug(f"Raw server response: {response}")
            
            decoded_response = response.decode()
            logging.debug(f"Decoded server response: {decoded_response}")
            
            try:
                return json.loads(decoded_response)
            except json.JSONDecodeError as json_err:
                logging.error(f"Failed to parse JSON: {json_err}")
                return {
                    "error": "Server response was not valid JSON",
                    "raw_response": decoded_response
                }
        
        except Exception as e:
            logging.error(f"Error in send_data: {e}")
            return {"error": str(e)}

    def get_session_data(self):
        self.audio.speak_openai(f"Hey {self.user_name}! On a scale of 1 to 10, how would you rate your performance in this session compared to previous sessions?")
        
        while True:
            performance_response = self.audio.listen_openai()
            # performance_response = "10"
            prompt = f"""
            The user responded with "{performance_response}" when asked to rate their performance on a scale of 1 to 10.
            Extract the numerical rating from this response. If the response is unclear, ambiguous, or doesn't contain a number between 1 and 10, return "unclear".
            Your response should be either a number between 1 and 10, or the word "unclear".
            """
            
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            
            interpreted_performance = completion.choices[0].message.content.strip()
            
            if interpreted_performance.isdigit() and 1 <= int(interpreted_performance) <= 10:
                performance = int(interpreted_performance)
                break
            elif interpreted_performance == "unclear":
                self.audio.speak_openai("I didn't quite catch that. Could you please give me a number between 1 and 10?")
            else:
                self.audio.speak_openai("I'm sorry, I need a number between 1 and 10. Could you try again?")
        
        return {'performance': performance}

    def generate_plots(self, stats):
        variables = ['skill', 'learning_rate', 'difficulty']
        plot_data = {}
        for var in variables:
            try:
                if var == 'learning_rate':
                    prior_shape = float(stats['prior_stats'][var][0])
                    prior_rate = float(stats['prior_stats'][var][1])
                    post_shape = float(stats['posterior_stats'][var][0])
                    post_rate = float(stats['posterior_stats'][var][1])
                    
                    prior_mean = prior_shape / prior_rate
                    post_mean = post_shape / post_rate
                    
                    x = np.linspace(0, max(prior_mean, post_mean) * 3, 1000)
                    
                    prior = scipy_stats.gamma.pdf(x, a=prior_shape, scale=1/prior_rate)
                    posterior = scipy_stats.gamma.pdf(x, a=post_shape, scale=1/post_rate)
                else:
                    prior_mean = float(stats['prior_stats'][var][0])
                    prior_std = float(stats['prior_stats'][var][1])
                    post_mean = float(stats['posterior_stats'][var][0])
                    post_std = float(stats['posterior_stats'][var][1])
                    
                    x = np.linspace(min(prior_mean, post_mean) - 3*max(prior_std, post_std),
                                    max(prior_mean, post_mean) + 3*max(prior_std, post_std),
                                    1000)
                    
                    prior = scipy_stats.norm.pdf(x, prior_mean, prior_std)
                    posterior = scipy_stats.norm.pdf(x, post_mean, post_std)
                
                plt.clf()
                plt.theme('dark')
                plt.plot(x, prior, label="Prior", color="cyan")
                plt.plot(x, posterior, label="Posterior", color="magenta")
                
                plt.vertical_line(prior_mean, color="cyan")
                plt.vertical_line(post_mean, color="magenta")
                
                shift_y = max(max(prior), max(posterior)) / 2
                plt.scatter([prior_mean, post_mean], [shift_y, shift_y], color="yellow")
                
                plt.title(f"{var.capitalize()} Distribution")
                plt.xlabel("Value")
                plt.ylabel("Density")
                plt.plotsize(100, 30)
                plt.show()
                
                if var == 'learning_rate':
                    print(f"Prior shape: {prior_shape:.2f}, Prior rate: {prior_rate:.2f}")
                    print(f"Posterior shape: {post_shape:.2f}, Posterior rate: {post_rate:.2f}")
                    print(f"Mean shift: {post_mean - prior_mean:.2f}")
                else:
                    print(f"Prior mean: {prior_mean:.2f}, Prior std: {prior_std:.2f}")
                    print(f"Posterior mean: {post_mean:.2f}, Posterior std: {post_std:.2f}")
                    print(f"Mean shift: {post_mean - prior_mean:.2f}")
                print()
                
                plot_data[var] = {
                    'prior_params': stats['prior_stats'][var],
                    'post_params': stats['posterior_stats'][var],
                    'prior_mean': prior_mean,
                    'post_mean': post_mean,
                    'x_range': [min(x), max(x)]
                }
            except Exception as e:
                logging.error(f"Error plotting {var}: {str(e)}")
                print(f"Error plotting {var}. See log for details.")
        
        return plot_data

    def generate_recommendation(self, plot_data, stats):

        prompt = f"""Using the following statistics on the user's skill progress, provide a critical analysis and recommendation:

        Prior Beliefs:
        - Skill: mean = {stats['prior_stats']['skill'][0]:.2f}, std = {stats['prior_stats']['skill'][1]:.2f}
        - Learning Rate: shape = {stats['prior_stats']['learning_rate'][0]:.2f}, rate = {stats['prior_stats']['learning_rate'][1]:.2f}
        - Difficulty: mean = {stats['prior_stats']['difficulty'][0]:.2f}, std = {stats['prior_stats']['difficulty'][1]:.2f}

        Updated Estimates:
        - Skill: mean = {stats['posterior_stats']['skill'][0]:.2f}, std = {stats['posterior_stats']['skill'][1]:.2f}
        - Learning Rate: shape = {stats['posterior_stats']['learning_rate'][0]:.2f}, rate = {stats['posterior_stats']['learning_rate'][1]:.2f}
        - Difficulty: mean = {stats['posterior_stats']['difficulty'][0]:.2f}, std = {stats['posterior_stats']['difficulty'][1]:.2f}

        Plot Data:
        {json.dumps(plot_data, indent=2)}

        A plot illustrating the shifts from prior to posterior distributions has been shown.

        Briefly summarize the following:
        1. Summarize the changes in skill, learning rate, and difficulty, referencing the plot.
        2. Evaluate whether these changes signify meaningful improvement.
        3. Describe what insights you have gained from the new data from the user.
        4. Offer a suggestion for improvement, mentioning any areas where progress is limited or skills may be overestimated.
        
        Address the user directly as "you" in your response. Be very concise.
        """

        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )

        return completion.choices[0].message.content.strip()

    def run(self):
        try:
            while True:
                session_data = self.get_session_data()
                results = self.send_data(session_data)
                results = json.loads(results)
                if 'error' in results:
                    self.audio.speak_openai(f"I'm sorry, there was an error communicating with the server. {results['error']}")
                    logging.error(f"Server communication error: {results}")
                    continue
                
                logging.debug(f"Received results from server: {results}")
                
                # Check if there's a date in the results and try to format it
                if 'date' in results:
                    try:
                        date_obj = datetime.strptime(results['date'], '%Y-%m-%d')
                        results['date'] = date_obj.strftime('%d/%m/%Y')
                    except ValueError:
                        logging.error(f"Invalid date format in results: {results['date']}")
                
                try:
                    plot_data = self.generate_plots(results)
                    logging.debug(f"Generated plot data: {plot_data}")
                except Exception as e:
                    logging.error(f"Error in generate_plots: {str(e)}")
                    self.audio.speak_openai("I'm sorry, there was an error generating the plots. Please check the logs for more information.")
                
                recommendation = self.generate_recommendation(plot_data, results)
                
                self.audio.speak_openai(f"Based on your input, here's an analysis of your progress.")
                self.audio.speak_openai(recommendation)
                
                self.audio.speak_openai("Would you like to add another session? Say 'yes' or 'no'.")
                choice = self.audio.listen_openai()
                if choice and 'no' in choice.lower():
                    break
        
        except KeyboardInterrupt:
            print("Client shutting down.")
        except Exception as e:
            logging.error(f"Unexpected error in run method: {str(e)}")
            self.audio.speak_openai("I'm sorry, an unexpected error occurred. Please check the logs for more information.")
        finally:
            self.client_socket.close()

if __name__ == "__main__":
    client = SkillProgressClient(user_name="Stranger")
    client.run()
