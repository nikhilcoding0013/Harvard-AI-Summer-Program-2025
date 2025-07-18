# Harvard-AI-Summer-Program-2025
Welcome to the repository for my projects and work in the Harvard AI Summer Bootcamp 2025, a five-day program exploring the fundamentals and frontiers of artificial intelligence. Each day focused on a core area in AI, combining lectures, hands-on labs, and optional homework challenges that I completed in full. This repository also includes my hackathon Cancer Detection Model, explained in detail.

## Day 1: Neural Networks and Optimization
On Day 1, we dove deep into the foundations of neural networks, including how models learn via forward and backpropagation. I gained a stronger understanding of the balance between overfitting and underfitting through the bias-variance tradeoff and understanding the influence of different choices in modeling. We also explored how gradient descent navigates the loss landscape and how techniques like dropout mitigate overfitting. In the coding lab, I built a binary classification model, exploring neural network architecture and math, and visualizing the decision boundary. The model performed with an evaluation accuracy of 100%. (Day_1_Neural_Networks.ipynb)

For homework, I explored hyperparameter tuning using a clothing classification dataset. I experimented with grid search and random search to optimize batch size, epoch, learning rate, number of neurons, activation function and optimizer — eventually reaching a validation accuracy of 87.34%, a significant improvement from the baseline. (Day_1_Hyperparameters_Optimization_HW.ipynb)

## Day 2: Transformers and Large Language Models
Day 2 was focused on the evolution and architecture of modern large language models. We started by exploring how models used to understand text using N-grams and RNNs, and then transitioned to how Transformers revolutionized that process. I learned about core components like attention mechanisms, positional encoding, and embeddings, and how they allow LLMs to understand context and relationships in language. We also broke down the history and structure of ChatGPT, which gave me a better understanding of the complexity behind conversational AI. In the coding lab, we compared the performances of an RNN (16% test accuracy) and a BERT transformer model (94% test accuracy) on a sentiment analysis dataset, demonstrating the superioirity of transformers over RNNs in language-based tasks. (Day_2_NLP_Classification_RNNs_and_BERT_Transformers.ipynb)

For homework #1 (Day_2_Fine_Tuning_HW.ipynb), I fine-tuned a pre-trained transformer model on a small dataset. This gave me hands-on experience with transfer learning and how fine-tuning works in practice. For the second assignment (Day_2_Mini_LLM_and_Quantization_HW.ipynb), I built a miniature LLM and explored the difference between single-head and multi-head attention. This helped me understand how attention layers weight different aspects of a sequence providing context to a model. I also experimented with quantization, which allowed me to reduce model size and memory usage while analyzing the slight trade-off in performance.

## Day 3: Convolutional Neural Networks & Computer Vision
On Day 3, we shifted our focus to computer vision and how convolutional neural networks (CNNs) process and understand images. I learned how convolutional layers parse visual data, and how operations like padding and pooling affect the dimensions and features passed through the network. We also discussed the overall architecture of CNNs and how they’re used in real-world applications—from facial recognition to autonomous driving. An interesting part of Day 3's lecture was a deep dive into Tesla Autopilot’s vision system, which showed how computer vision models are deployed at scale in high-stakes environments.

During the lab session, I trained a CNN to perform handwriting image classification and experimented with data augmentation to improve generalization. Working through different augmentation techniques helped me understand how to combat overfitting and make models more robust to variations in input. Day 3 gave me a much clearer understanding of why computer vision models are essential to modern AI and how they power many of today’s impactful technologies. (Day_3_Computer_Vision_Handwriting_Recognition_with_CNN.ipynb)

## Day 4: Reinforcement Learning
Day 4 introduced us to the world of reinforcement learning, where agents interact with an environment and learn optimal behaviors through a system of rewards and penalties. Toward the start of Day 4, we discussed the Markov decision process as well as techniques such as policy iteration, policy evaluation, and value iteration in addition to time constraints. We also focused in on modern approaches like Q-learning and the math behind them. Day 4 also discussed the difference between model-based and model-free learning, and how these paradigms are applied in real-world systems like robotics, game-playing AI, and autonomous driving.

The Day 4 lab (Day_4_Reinforced_Learning_Multi_Armed_Bandit;_Random,_Epsilon_Greedy,_UCB.ipynb) went into detail on the exploration-explotation dilemma, whether the agent should continue exploring or choose the most rewarding outcome. In the lab, I implemented and evaluated three agents in a multi-armed bandit setting: a random agent, an epsilon-greedy agent that balances exploration and exploitation, and a UCB (Upper Confidence Bound) agent that uses uncertainty to guide exploration. Each strategy was tested over time, and I tracked total cumulative rewards to compare their learning efficiency. As shown in the results below (Day_4_Total_Reward_Over_Time_for_Different_Agents.jpg), the UCB agent significantly outperformed both the random and epsilon-greedy agents, demonstrating how intelligent exploration strategies can drastically improve long-term reward.

<img width="1481" height="464" alt="Screenshot 2025-07-17 142700" src="https://github.com/user-attachments/assets/17a50abc-c26b-4ef3-9755-994ffec55e64" />

## Day 5: Hackathon
The final day of the program was a timed hackathon, where each participant selected one of eleven open-ended machine learning challenges. I chose a project titled "Classifying Benign vs. Malignant Breast Tumors Using Deep Learning" — a task that hit close to home given cancer’s impact on my own family. The challenge involved filling in major gaps in a partially built framework and then taking the model beyond the baseline by applying concepts we had learned throughout the week.

The project included two architectures: a custom CNN and a pretrained ResNet. The more advanced ResNet model performed with a maximum validation accuracy of 80% straight off the bat while the CNN's maximum was 50%. I quickly identified clear signs of overfitting, high training accuracy paired with much lower validation accuracy, and used data augmentation to address this. Drawing on what I had learned about transfer learning and fine-tuning from the transformer module, I focused my efforts on the ResNet, believing it had higher potential. I experimented with the architecture by adding complexity: more linear layers with dropout layers to reduce overfitting, and activation function adjustments.
These strategies raised the maximum evaluation accuracy of the ResNet model from 80% to 85%.

### Baseline:

<img width="960" height="480" alt="Untitled presentation" src="https://github.com/user-attachments/assets/c0260dfb-d29e-48b8-adf5-1affb8d85b54" />

### 85% maximum validation accuracy:

<img width="960" height="480" alt="Untitled presentation (1)" src="https://github.com/user-attachments/assets/e686c6d2-e908-406b-8c69-387c53a8f881" />

Given more time, I planned to run a hyperparameter grid search. But with only 2.5 hours, I pivoted to a random search strategy — though I was ultimately cut short on time before I could run it. Regardless, I presented my work effectively to the class, using visualizations to explain my modeling decisions, observations, and results.

After the hackathon ended, I decided to revisit the project on my own given that Cancer Detection is a personal challenge to me. Motivated by the real-world stakes, I ran a random search selecting 6 at a time on 144 possible hyperparameter combinations. After multiple iterations, I identified a high-performing combination:
224 neurons, 4 epochs, learning rate of 0.0001, and batch size of 64.
When I trained the model with this configuration, the results were impressive. The validation accuracy after 1 epoch when training for 4 jumped to 95%, far exceeding all prior results.

### 95% maximum validation accuracy:

<img width="960" height="480" alt="Untitled presentation (2)" src="https://github.com/user-attachments/assets/6f012470-ef3c-4ce2-afc0-ef37639d6c25" />

### Conclusion
**This project and the Harvard AI Bootcamp helped me bring together everything I had learned over the week — from computer vision architectures and transfer learning, to regularization and hyperparameter optimization. More importantly, it showed me how technical skills can be used to work on problems that matter personally and globally.**
