using UnityEngine;

public class MeshUndulator : MonoBehaviour
{
    public MeshFilter meshFilter;
    public float amplitude = 1f;
    public float frequency = 1f;
    public float speed = 1f;

    private Vector3[] originalVertices;
    private Vector3[] modifiedVertices;

    private void Start()
    {
        // Check if a mesh filter is assigned
        if (meshFilter == null)
        {
            Debug.LogError("Mesh filter not assigned!");
            return;
        }

        // Get the original vertices of the mesh
        originalVertices = meshFilter.mesh.vertices;

        // Create a new array to store the modified vertices
        modifiedVertices = new Vector3[originalVertices.Length];
    }

    private void Update()
    {
        // Iterate through each vertex of the mesh
        for (int i = 0; i < originalVertices.Length; i++)
        {
            // Calculate the new Y position based on a sine wave
            float newY = originalVertices[i].y + amplitude * Mathf.Sin(Time.time * speed + originalVertices[i].x * frequency);

            // Update the modified vertex position
            modifiedVertices[i] = new Vector3(originalVertices[i].x, newY, originalVertices[i].z);
        }

        // Update the mesh with the modified vertices
        meshFilter.mesh.vertices = modifiedVertices;
        meshFilter.mesh.RecalculateNormals();
    }
}
