    // static/js/confirm.js

    const descriptionElement = document.getElementById('checklist-description');
    const checklistSelect = document.getElementById('chosen_checklist');

    /**
     * Fetches and displays the description for the selected checklist.
     * @param {string} municipality - The current municipality.
     * @param {string} checklistName - The name of the selected checklist.
     */
    async function fetchDescription(municipality, checklistName) {
        if (!checklistName) {
            descriptionElement.textContent = 'Please select a checklist to see its description.';
            return;
        }

        descriptionElement.textContent = 'Loading description...'; // Show loading state

        try {
            // Construct the API URL dynamically
            const apiUrl = `/api/checklist_description/${encodeURIComponent(municipality)}/${encodeURIComponent(checklistName)}`;
            const response = await fetch(apiUrl);

            if (!response.ok) {
                // Handle HTTP errors (e.g., 404 Not Found, 500 Server Error)
                const errorData = await response.json().catch(() => ({})); // Try to parse error JSON, default to empty object
                console.error(`Error fetching description: ${response.status} ${response.statusText}`, errorData);
                descriptionElement.textContent = `Error loading description: ${errorData.error || response.statusText}`;
                descriptionElement.classList.add('error');
                descriptionElement.classList.remove('warning');
                return;
            }

            const data = await response.json();

            if (data.error) {
                 // Handle errors reported in the JSON payload
                console.error('API Error:', data.error);
                descriptionElement.textContent = `Error: ${data.error}`;
                descriptionElement.classList.add('error');
                 descriptionElement.classList.remove('warning');
            } else {
                // Display the fetched description
                descriptionElement.textContent = data.description || 'No description available.';
                descriptionElement.classList.remove('error', 'warning'); // Clear error/warning states
            }

        } catch (error) {
            // Handle network errors or issues parsing JSON
            console.error('Network or parsing error:', error);
            descriptionElement.textContent = 'Failed to load description due to a network or server issue.';
            descriptionElement.classList.add('error');
            descriptionElement.classList.remove('warning');
        }
    }

    // Add event listener to the select dropdown
    if (checklistSelect) {
        checklistSelect.addEventListener('change', (event) => {
            // Get the currently selected municipality (assuming it's available globally or passed)
            // Note: 'currentMunicipality' should be defined in the HTML script block
            if (typeof currentMunicipality !== 'undefined') {
                 fetchDescription(currentMunicipality, event.target.value);
            } else {
                console.error("Municipality variable 'currentMunicipality' is not defined.");
                descriptionElement.textContent = 'Error: Municipality context is missing.';
            }
        });
    } else {
        console.error("Element with ID 'chosen_checklist' not found.");
    }
    