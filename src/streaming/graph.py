from collections import defaultdict, deque
import time
from typing import Dict, List, Set, Tuple, Any, Optional

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class FraudRingGraphEngine:
    """
    In-memory bipartite entity graph tracking relationships between cards and devices.
    Identifies multi-card device sharing rings, botnet attacks, and synthetic identity syndicates
    within a rolling event-time window (default: 24 hours).
    """

    def __init__(self, ttl_seconds: float = 86400.0, ring_threshold: int = 3):
        self.ttl_seconds = ttl_seconds
        self.ring_threshold = ring_threshold
        # Bipartite adjacency mapping
        self.cards_to_devices: Dict[str, Dict[str, float]] = defaultdict(dict)  # card_id -> {device_id: timestamp}
        self.devices_to_cards: Dict[str, Dict[str, float]] = defaultdict(dict)  # device_id -> {card_id: timestamp}
        # Edge history queue for efficient TTL eviction: (timestamp, card_id, device_id)
        self.edge_history = deque()

    def prune_stale_edges(self, current_time: float) -> None:
        """Evicts relationships older than the rolling TTL window."""
        cutoff = current_time - self.ttl_seconds
        while self.edge_history and self.edge_history[0][0] < cutoff:
            t, card, device = self.edge_history.popleft()
            # If the recorded timestamp matches the latest timestamp in the map, delete edge
            if card in self.cards_to_devices and self.cards_to_devices[card].get(device) == t:
                del self.cards_to_devices[card][device]
                if not self.cards_to_devices[card]:
                    del self.cards_to_devices[card]

            if device in self.devices_to_cards and self.devices_to_cards[device].get(card) == t:
                del self.devices_to_cards[device][card]
                if not self.devices_to_cards[device]:
                    del self.devices_to_cards[device]

    def record_interaction(self, card_id: str, device_id: Optional[str], timestamp: float) -> Tuple[int, int, int, bool]:
        """
        Records a transaction interaction between card and device,
        returns (device_card_count, card_device_count, ring_cluster_size, is_fraud_ring_suspected).
        """
        if not device_id or not card_id:
            return 1, 1, 1, False

        self.prune_stale_edges(timestamp)

        # Update edges
        self.cards_to_devices[card_id][device_id] = timestamp
        self.devices_to_cards[device_id][card_id] = timestamp
        self.edge_history.append((timestamp, card_id, device_id))

        # Calculate metrics
        device_card_count = len(self.devices_to_cards[device_id])
        card_device_count = len(self.cards_to_devices[card_id])

        # Compute connected cluster
        cluster = self.get_connected_cluster(card_id)
        cluster_size = cluster["cluster_size"]

        is_ring = (device_card_count >= self.ring_threshold) or (len(cluster["cards"]) >= self.ring_threshold)
        return device_card_count, card_device_count, cluster_size, is_ring

    def get_connected_cluster(self, start_card_id: str) -> Dict[str, Any]:
        """
        Performs breadth-first search (BFS) over bipartite graph to find
        all connected cards and devices in the syndicate component.
        """
        if start_card_id not in self.cards_to_devices:
            return {"cluster_size": 1, "cards": [start_card_id], "devices": []}

        visited_cards: Set[str] = set()
        visited_devices: Set[str] = set()
        card_queue = deque([start_card_id])

        while card_queue:
            curr_card = card_queue.popleft()
            if curr_card in visited_cards:
                continue
            visited_cards.add(curr_card)

            for dev in self.cards_to_devices.get(curr_card, {}):
                if dev not in visited_devices:
                    visited_devices.add(dev)
                    for next_card in self.devices_to_cards.get(dev, {}):
                        if next_card not in visited_cards:
                            card_queue.append(next_card)

        total_nodes = len(visited_cards) + len(visited_devices)
        return {
            "cluster_size": total_nodes,
            "cards": list(visited_cards),
            "devices": list(visited_devices),
        }

    def get_detected_fraud_rings(self) -> List[Dict[str, Any]]:
        """Finds all active clusters with multi-card device sharing."""
        rings = []
        visited_devices = set()

        for dev, cards_dict in self.devices_to_cards.items():
            if dev in visited_devices or len(cards_dict) < self.ring_threshold:
                continue

            first_card = next(iter(cards_dict.keys()))
            cluster = self.get_connected_cluster(first_card)
            for d in cluster["devices"]:
                visited_devices.add(d)

            rings.append({
                "anchor_device": dev,
                "card_count": len(cluster["cards"]),
                "device_count": len(cluster["devices"]),
                "cards": cluster["cards"],
                "devices": cluster["devices"],
                "total_entities": cluster["cluster_size"],
            })

        rings.sort(key=lambda x: x["card_count"], reverse=True)
        return rings

    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Generates 2D node coordinates and edge traces for interactive Plotly rendering.
        """
        if not HAS_NETWORKX or not self.devices_to_cards:
            return {"nodes": [], "edges": []}

        G = nx.Graph()
        for dev, cards in self.devices_to_cards.items():
            G.add_node(dev, type="device", label=dev)
            for card in cards:
                G.add_node(card, type="card", label=card)
                G.add_edge(card, dev)

        # Compute 2D layout
        pos = nx.spring_layout(G, k=0.5, seed=42)

        node_data = []
        for node in G.nodes():
            x, y = pos[node]
            ntype = G.nodes[node]["type"]
            degree = G.degree[node]
            is_suspicious = (ntype == "device" and degree >= self.ring_threshold) or (ntype == "card" and degree >= 2)
            node_data.append({
                "id": node,
                "x": float(x),
                "y": float(y),
                "type": ntype,
                "degree": degree,
                "is_suspicious": is_suspicious,
            })

        edge_data = []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_data.append({
                "source": u,
                "target": v,
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
            })

        return {"nodes": node_data, "edges": edge_data}

    def reset(self) -> None:
        self.cards_to_devices.clear()
        self.devices_to_cards.clear()
        self.edge_history.clear()
