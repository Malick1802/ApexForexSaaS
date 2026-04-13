
import yaml
import os

CONFIG_PATH = "config.yaml"

def disable_jpy():
    if not os.path.exists(CONFIG_PATH):
        print(f"Config not found: {CONFIG_PATH}")
        return

    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
            
        pairs_config = config.get('currency_pairs', {})
        removed_count = 0
        removed_list = []
        
        for category in ['majors', 'minors', 'crosses']:
            if category in pairs_config:
                original_list = pairs_config[category]
                new_list = []
                for item in original_list:
                    symbol = item.get('symbol', '')
                    if 'JPY' in symbol:
                        print(f"Removing {symbol} from {category}")
                        removed_list.append(symbol)
                        removed_count += 1
                    else:
                        new_list.append(item)
                pairs_config[category] = new_list
                
        config['currency_pairs'] = pairs_config
        
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=False)
            
        print(f"Successfully removed {removed_count} JPY pairs.")
        print(f"Removed: {', '.join(removed_list)}")
        
    except Exception as e:
        print(f"Error updating config: {e}")

if __name__ == "__main__":
    disable_jpy()
