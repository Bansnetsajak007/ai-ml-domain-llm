async function loadDashboard() {
    try {
        const response = await fetch('/api/resources');
        const resources = await response.json();
        const tbody = document.getElementById('resourceTableBody');
        tbody.innerHTML = '';

        resources.forEach(r => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${r.title}</td>
                <td>${r.type}</td>
                <td>${Array.isArray(r.authors) ? r.authors.join(', ') : r.authors}</td>
                <td>${r.year}</td>
                <td>${r.source}</td>
                <td>${r.added_by}</td>
            `;
            tbody.appendChild(row);
        });
    } catch (error) {
        console.error('Error loading dashboard:', error);
    }
}

document.getElementById('addResourceForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('submitBtn');
    const msg = document.getElementById('statusMessage');
    const warningDiv = document.getElementById('duplicateWarning');

    // Reset UI
    btn.disabled = true;
    msg.textContent = 'Checking...';
    warningDiv.classList.add('hidden');
    msg.className = '';

    const collectorName = document.getElementById('collectorName').value;
    const jsonText = document.getElementById('resourceJson').value;

    // Validate JSON
    let resourceData;
    try {
        resourceData = JSON.parse(jsonText);
    } catch (err) {
        msg.textContent = 'Invalid JSON: ' + err.message;
        msg.style.color = 'red';
        btn.disabled = false;
        return;
    }

    const payload = {
        collector_name: collectorName,
        resource_data: resourceData
    };

    try {
        const response = await fetch('/api/resources', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (response.ok) {
            msg.textContent = result.message || 'Saved successfully!';
            msg.style.color = 'green';
            document.getElementById('addResourceForm').reset();
        } else if (response.status === 409) {
            // Duplicate or Warning
            msg.textContent = 'Warning: Possible Duplicate';
            msg.style.color = 'orange';

            warningDiv.classList.remove('hidden');
            document.getElementById('geminiExplanation').textContent = result.explanation || "Duplicate detected.";
            document.getElementById('existingResource').textContent = JSON.stringify(result.existing || {}, null, 2);

            // Handle actions
            document.getElementById('cancelBtn').onclick = () => {
                warningDiv.classList.add('hidden');
                msg.textContent = 'Cancelled.';
                btn.disabled = false;
            };

            document.getElementById('forceSaveBtn').onclick = async () => {
                await forceSave(payload);
            };
        } else {
            msg.textContent = 'Error: ' + result.error;
            msg.style.color = 'red';
        }
    } catch (error) {
        msg.textContent = 'Network Error';
        console.error(error);
    } finally {
        if (document.getElementById('duplicateWarning').classList.contains('hidden')) {
            btn.disabled = false;
        }
    }
});

async function forceSave(payload) {
    const msg = document.getElementById('statusMessage');
    payload.force_save = true;

    try {
        const response = await fetch('/api/resources', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            msg.textContent = 'Force saved successfully!';
            msg.style.color = 'green';
            document.getElementById('duplicateWarning').classList.add('hidden');
            document.getElementById('addResourceForm').reset();
            document.getElementById('submitBtn').disabled = false;
        } else {
            const res = await response.json();
            alert('Failed to force save: ' + res.error);
        }
    } catch (err) {
        alert('Error: ' + err.message);
    }
}

// Delete Functionality
const deleteModal = document.getElementById('deleteModal');
if (deleteModal) {
    document.getElementById('openDeleteModalBtn').onclick = () => {
        deleteModal.classList.remove('hidden');
    };

    document.getElementById('closeDeleteModalBtn').onclick = () => {
        deleteModal.classList.add('hidden');
    };

    document.getElementById('deleteAllBtn').onclick = async () => {
        if (confirm("Are you SURE you want to delete ALL resources? This cannot be undone.")) {
            try {
                const response = await fetch('/api/resources?all=true', { method: 'DELETE' });
                const res = await response.json();
                alert(res.message);
                deleteModal.classList.add('hidden');
                loadDashboard(); // Refresh table
            } catch (error) {
                alert("Error deleting: " + error);
            }
        }
    };

    document.getElementById('deleteCollectorBtn').onclick = async () => {
        const name = prompt("Enter the Collector Name to delete resources for:");
        if (name) {
            if (confirm(`Are you SURE you want to delete all resources added by '${name}'?`)) {
                try {
                    const response = await fetch(`/api/resources?collector=${encodeURIComponent(name)}`, { method: 'DELETE' });
                    const res = await response.json();
                    alert(res.message);
                    deleteModal.classList.add('hidden');
                    loadDashboard(); // Refresh table
                } catch (error) {
                    alert("Error deleting: " + error);
                }
            }
        }
    };
}
