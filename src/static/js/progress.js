    // static/js/progress.js

    document.addEventListener('DOMContentLoaded', () => {
        const progressBar = document.getElementById('progress-bar');
        const progressMessage = document.getElementById('progress-message');
        const etaMessage = document.getElementById('eta-message');
        const resultsSection = document.getElementById('results-section');
        const resultsTbody = document.getElementById('results-tbody');
        const downloadLinksDiv = document.getElementById('download-links');
        const errorSection = document.getElementById('error-section');
        const errorMessage = document.getElementById('error-message');
        const spinner = document.getElementById('spinner');

        // --- Helper Function to Format ETA ---
        function formatETA(seconds) {
            if (seconds < 0 || seconds === null || typeof seconds === 'undefined') {
                return 'Calculating...';
            }
            if (seconds === 0) {
                return 'Almost done...';
            }
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = seconds % 60;
            let etaString = '';
            if (minutes > 0) {
                etaString += `${minutes} minute${minutes > 1 ? 's' : ''}`;
            }
            if (remainingSeconds > 0) {
                if (etaString) etaString += ', ';
                etaString += `${remainingSeconds} second${remainingSeconds > 1 ? 's' : ''}`;
            }
            return `Approx. ${etaString} remaining`;
        }

        // --- Helper Function to Populate Table ---
        function populateTable(results) {
            resultsTbody.innerHTML = ''; // Clear previous results
            if (!results || results.length === 0) {
                resultsTbody.innerHTML = '<tr><td colspan="4">No results generated.</td></tr>';
                return;
            }
            results.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${row['Numero Punto'] || 'N/A'}</td>
                    <td>${row['Testo Punto'] || ''}</td>
                    <td>${row['Risposta Semplice'] || 'N/A'}</td>
                    <td><pre>${row['Risposta LLM Completa'] || ''}</pre></td>
                `;
                // Add styling for ERROR rows if needed
                if (row['Risposta Semplice'] === 'ERROR') {
                    tr.style.backgroundColor = '#f8d7da'; // Light red background for errors
                    tr.style.color = '#721c24';
                }
                resultsTbody.appendChild(tr);
            });
        }

        // --- Helper Function to Create Download Links ---
         function createDownloadLinks(filenames) {
            downloadLinksDiv.innerHTML = '<strong>Download Results:</strong> '; // Clear previous links
            const formats = { // Map extension to display name and order
                 csv: 'CSV',
                 json: 'JSON',
                 xlsx: 'XLSX'
             };
             const orderedExtensions = ['csv', 'json', 'xlsx']; // Desired order

             orderedExtensions.forEach(ext => {
                 if (filenames[ext]) {
                     const filename = filenames[ext];
                     const formatName = formats[ext];
                     const link = document.createElement('a');
                     link.href = `/download/${encodeURIComponent(filename)}`;
                     link.textContent = formatName;
                     link.style.marginLeft = '10px';
                     link.style.padding = '5px 10px';
                     link.style.backgroundColor = '#007bff';
                     link.style.color = 'white';
                     link.style.borderRadius = '4px';
                     link.style.textDecoration = 'none';
                     link.setAttribute('download', filename); // Suggest filename
                     downloadLinksDiv.appendChild(link);
                 }
             });
             if (downloadLinksDiv.children.length <= 1) { // Only the initial text is present
                  downloadLinksDiv.innerHTML += ' No download files available.';
             }
         }


        // --- Connect to SSE Stream ---
        const eventSource = new EventSource('/stream'); // Connect to the stream endpoint

        eventSource.onopen = function() {
            console.log("SSE Connection opened.");
            progressMessage.textContent = "Connection established. Starting analysis...";
            spinner.style.display = 'block'; // Show spinner
        };

        // --- Handle Progress Events ---
        eventSource.addEventListener('progress', function(event) {
            console.log("Progress event received:", event.data);
            const data = JSON.parse(event.data);

            const percentage = data.total > 0 ? Math.round((data.current / data.total) * 100) : 0;
            progressBar.style.width = percentage + '%';
            progressBar.textContent = percentage + '%';
            progressMessage.textContent = data.message || `Processing point ${data.current}/${data.total}...`;
            etaMessage.textContent = `ETA: ${formatETA(data.eta_seconds)}`;
            spinner.style.display = 'block'; // Keep spinner visible
        });

         // --- Handle Point Error Events ---
         eventSource.addEventListener('point_error', function(event) {
            console.warn("Point error event received:", event.data);
            const data = JSON.parse(event.data);
            // Optionally display a temporary warning or log it more visibly
            // progressMessage.textContent = `Warning: Error on point ${data.point_num}. Continuing...`;
            // You could add a list of points with errors
         });


        // --- Handle Completion Event ---
        eventSource.addEventListener('complete', function(event) {
            console.log("Complete event received:", event.data);
            const data = JSON.parse(event.data);

            progressBar.style.width = '100%';
            progressBar.textContent = '100%';
            progressMessage.textContent = `Completed: ${data.message}`;
            etaMessage.textContent = 'ETA: Done!';
            spinner.style.display = 'none'; // Hide spinner

            if (data.status === 'success') {
                populateTable(data.results || []); // Populate table with results data
                createDownloadLinks(data.download_filenames || {}); // Create download links
                resultsSection.classList.remove('hidden'); // Show results
            } else {
                // Handle completion status that isn't 'success' if needed
                errorMessage.textContent = data.message || 'Task completed with non-success status.';
                errorSection.classList.remove('hidden');
            }

            eventSource.close(); // Close the connection
            console.log("SSE Connection closed on completion.");
        });

        // --- Handle Error Events ---
        eventSource.onerror = function(event) {
            console.error("SSE Error occurred:", event);
            let message = "An unknown error occurred with the progress stream.";
             // Attempt to parse error data if available (though onerror often doesn't have custom data)
             // Check if it's a custom error event first
             if (event.type === 'error' && event.data) {
                 try {
                     const data = JSON.parse(event.data);
                     message = `Error: ${data.message || 'Unknown server error.'}`;
                 } catch (e) {
                     console.warn("Could not parse error event data:", event.data);
                     message = "Received an unparseable error from the server.";
                 }
             } else if (eventSource.readyState === EventSource.CLOSED) {
                 message = "Connection to server lost or closed unexpectedly.";
                 // Don't show error section if closed normally after completion
                 if (!resultsSection.classList.contains('hidden')) {
                     return; // Already completed successfully
                 }
             } else {
                 message = "Connection error. Please check your network or try again.";
             }


            progressMessage.textContent = "Error!";
            etaMessage.textContent = "ETA: Failed";
            errorMessage.textContent = message;
            errorSection.classList.remove('hidden'); // Show error section
            spinner.style.display = 'none'; // Hide spinner

            eventSource.close(); // Close the connection
            console.log("SSE Connection closed due to error.");
        };

        // Optional: Handle window close to attempt closing SSE connection
        window.addEventListener('beforeunload', () => {
             if (eventSource && eventSource.readyState !== EventSource.CLOSED) {
                 console.log("Closing SSE connection on page unload.");
                 eventSource.close();
             }
         });

    });
    