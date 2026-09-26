// Drop this file into:  Assets/_Project/Editor/CharacterModelImporter.cs
//
// Fixes: "Not allowed to access vertices on mesh 'Body' (isReadable is false)"
// BlockyBody.MeasureBones reads mesh.vertices, which Unity only allows when the
// model was imported with Read/Write enabled. This importer turns Read/Write on
// for every model under Assets/_Project, and re-imports any model that was
// imported before this script existed.
using UnityEditor;
using UnityEngine;

public class CharacterModelImporter : AssetPostprocessor
{
    const string Root = "Assets/_Project";

    void OnPreprocessModel()
    {
        if (!assetPath.StartsWith(Root)) return;
        if (assetImporter is ModelImporter importer && !importer.isReadable)
            importer.isReadable = true;
    }

    // One-time fix-up for models that were imported before this script was added.
    [InitializeOnLoadMethod]
    static void FixExistingModels()
    {
        EditorApplication.delayCall += () =>
        {
            foreach (var guid in AssetDatabase.FindAssets("t:Model", new[] { Root }))
            {
                var path = AssetDatabase.GUIDToAssetPath(guid);
                if (AssetImporter.GetAtPath(path) is ModelImporter mi && !mi.isReadable)
                {
                    mi.isReadable = true;
                    mi.SaveAndReimport();
                    Debug.Log($"[CharacterModelImporter] Enabled Read/Write on {path}");
                }
            }
        };
    }
}
