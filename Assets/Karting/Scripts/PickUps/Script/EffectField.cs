using UnityEngine;
using KartGame.KartSystems;

public class EffectField : MonoBehaviour
{
    [SerializeField]
    [Range(0.0f, 2.0f)]
    [Tooltip("Modify the Speed of the Cars within.\n1.0: Normal.\n>1.0: Fast(Down Hill).\n<1.0: Slow(Up Hill)\nThe Color doesn't show up in actual game.")]
    private float SpeedModifier;

    public void OnTriggerEnter(Collider other)
    {
        if (other.transform.parent.TryGetComponent(out ArcadeKart kart))
            PickUpManager.instance.EnterEffect(SpeedModifier, kart.MyID);
    }

    public void OnTriggerExit(Collider other)
    {
        if (other.transform.parent.TryGetComponent(out ArcadeKart kart))
            PickUpManager.instance.LeaveEffect(kart.MyID);
    }

    private void OnDrawGizmos()
    {
        if (SpeedModifier > 1.0f)
            Gizmos.color = new Color(0, 0, SpeedModifier - 1.0f, 0.5f);
        else
            Gizmos.color = new Color(1.0f - SpeedModifier, 0, 0, 0.5f);

        Gizmos.matrix = this.transform.localToWorldMatrix;
        Gizmos.DrawCube(Vector3.zero, Vector3.one);
    }
}