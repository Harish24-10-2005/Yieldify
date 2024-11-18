document.addEventListener('DOMContentLoaded', function () {
  const chatContainer = document.getElementById('chat-messages');
  const userInputField = document.getElementById('user-input');
  const inputForm = document.getElementById('input-form');

  // Function to display bot message in chat
  function displayBotMessage(message) {
      const botBubble = document.createElement('div');
      botBubble.classList.add('bubble', 'bot-bubble');
      botBubble.textContent = message;
      chatContainer.appendChild(botBubble);
      // Scroll to bottom of chat
      chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  // Function to handle user input
  async function handleUserInput(input) {
      // Display user message in chat
      const userBubble = document.createElement('div');
      userBubble.classList.add('bubble', 'user-bubble');
      userBubble.textContent = input;
      chatContainer.appendChild(userBubble);
      // Scroll to bottom of chat
      chatContainer.scrollTop = chatContainer.scrollHeight;

      // Call API to get bot response
      const response = await fetchBotResponse(input);
      displayBotMessage(response);
  }

  // Function to fetch bot response from API
  async function fetchBotResponse(input) {
      const url = "https://api.gooey.ai/v2/video-bots/";
      const payload = JSON.stringify({
          "input_prompt": input
      });
      const headers = {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer sk-xJLOfOHXMFFAI6MOTHMwTPVozFo5Q3trH2mahV90jdO2Pjtm',
          'api-key': 'sk-xJLOfOHXMFFAI6MOTHMwTPVozFo5Q3trH2mahV90jdO2Pjtm'
      };

      try {
          const response = await fetch(url, {
              method: 'POST',
              headers: headers,
              body: payload
          });

          const data = await response.json();
          const rawOutputText = data.output.raw_output_text;

          // Join multiple responses into a single string
          const responseText = rawOutputText.join('\n');

          return responseText;
      } catch (error) {
          return 'Error: ' + error.message;
      }
  }

  // Handle form submission
  inputForm.addEventListener('submit', async function (e) {
      e.preventDefault();
      const userInput = userInputField.value.trim();
      if (userInput !== '') {
          await handleUserInput(userInput);
          userInputField.value = '';
      }
  });

  // Initial greeting from the bot
  displayBotMessage("Hi there! How can I assist you?");
});
