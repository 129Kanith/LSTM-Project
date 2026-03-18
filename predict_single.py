import argparse
import sys
from inference import threatInference

def main():
    parser = argparse.ArgumentParser(description="Predict attack type for a single network event")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--protocol_type", type=str, default="tcp", help="tcp, udp, or icmp")
    parser.add_argument("--src_bytes", type=float, default=0.0)
    parser.add_argument("--dst_bytes", type=float, default=0.0)
    parser.add_argument("--failed_logins", type=int, default=0)
    parser.add_argument("--logged_in", type=int, default=0)
    parser.add_argument("--count", type=float, default=0.0)
    parser.add_argument("--srv_count", type=float, default=0.0)
    parser.add_argument("--serror_rate", type=float, default=0.0)
    parser.add_argument("--srv_serror_rate", type=float, default=0.0)

    args = parser.parse_args()
    
    # Pack into dictionary
    packet_data = vars(args)
    
    try:
        # Initialize engine
        engine = threatInference()
        
        # Predict
        result = engine.predict_single(packet_data)
        
        print("\n" + "="*35)
        print("   CYBER THREAT DETECTION RESULT")
        print("="*35)
        print(f"Predicted Class: {result['prediction']}")
        print(f"Confidence:      {result['confidence']:.2%}")
        print("-" * 35)
        print("Top Probabilities:")
        top_probs = sorted(result['distribution'].items(), key=lambda x: x[1], reverse=True)[:3]
        for label, prob in top_probs:
            print(f" - {label:<12}: {prob:.2%}")
        print("="*35 + "\n")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
