const API_URL = "http://localhost:8000";

export async function fetchPreds(){
    const response = await fetch(`${API_URL}/api/predictions`);
    if (!response.ok) {
        throw new Error('Failed to fetch predictions');
    }
    return response.json();
}

export async function createPred(file) {
    const formData = new FormData()
    formData.append("file", file);
    const response = await fetch(`${API_URL}/api/uploadfile`, {
        method: 'POST',
        body: formData
    });
    if (!response.ok) {
        throw new Error('Failed to create prediction');
    }
    return response.json();
}