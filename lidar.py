import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat
import time

class GenesisLidar:
    """
    Genesisシミュレータ用のLIDARセンサクラス。
    数学的なレイ-形状交差判定を用いて実装されています。
    可視化には scene.draw_debug_line を使用します。
    """
    def __init__(
        self,
        scene: gs.Scene,
        robot: gs.engine.entities.rigid_entity.rigid_entity.RigidEntity,
        obstacle_entities: list,
        num_rays: int = 180,
        ray_length: float = 1.0,
        ray_start_angle: float = -math.pi,
        ray_end_angle: float = math.pi,
        height_offset: float = 0.1,
        visualize: bool = True
    ):
        """
        Args:
            scene (gs.Scene): シミュレーションシーンのインスタンス
            robot (gs.Entity): LIDARがアタッチされるロボットのエンティティ
            obstacle_entities (list): 衝突判定の対象となる障害物エンティティのリスト
            num_rays (int): レイの数
            ray_length (float): レイの最大長
            ray_start_angle (float): レイの開始角度 (rad)
            ray_end_angle (float): レイの終了角度 (rad)
            height_offset (float): ロボットの基底からのLIDARの高さオフセット
            visualize (bool): LIDARのレイを可視化するかどうか
        """
        self.scene = scene
        self.robot = robot
        self.obstacle_entities = obstacle_entities
        self.num_envs = self.scene.n_envs
        self.device = gs.device
        
        self.num_rays = num_rays
        self.ray_length = ray_length
        self.height_offset = height_offset
        self.visualize = visualize

        # ローカル座標系でのレイの方向ベクトルを事前に計算
        angles = torch.linspace(ray_start_angle, ray_end_angle, num_rays, device=self.device)
        self.local_ray_directions = torch.stack([
            torch.cos(angles),
            torch.sin(angles),
            torch.zeros_like(angles)
        ], dim=-1) # Shape: (num_rays, 3)

        if self.visualize:
            # 可視化対象の環境インデックス (最初の環境のみ)
            self.vis_env_idx = 0

    def scan(self):
        """
        LIDARスキャンを実行し、障害物までの距離を返す。
        Returns:
            torch.Tensor: 各レイの距離データ (num_envs, num_rays)
        """
        # ロボットの現在の位置と向きを取得
        base_pos = self.robot.get_pos()
        base_quat = self.robot.get_quat()

        # レイの始点を計算
        ray_origins = base_pos.unsqueeze(1) + torch.tensor([0, 0, self.height_offset], device=self.device)

        # ローカルのレイ方向をワールド座標系に変換
        world_ray_directions = transform_by_quat(
            self.local_ray_directions.expand(self.num_envs, -1, -1),
            base_quat.unsqueeze(1).expand(-1, self.num_rays, -1)
        )

        # 距離を最大長で初期化
        distances = torch.full((self.num_envs, self.num_rays), self.ray_length, device=self.device)

        # 各障害物との交差判定
        for obstacle in self.obstacle_entities:
            # 障害物の種類に応じて判定メソッドを呼び出す
            if isinstance(obstacle.morph, gs.morphs.Box):
                hit_distances = self._ray_intersect_box(ray_origins, world_ray_directions, obstacle)
            elif isinstance(obstacle.morph, gs.morphs.Sphere):
                hit_distances = self._ray_intersect_sphere(ray_origins, world_ray_directions, obstacle)
            elif isinstance(obstacle.morph, gs.morphs.Cylinder):
                 hit_distances = self._ray_intersect_cylinder(ray_origins, world_ray_directions, obstacle)
            else:
                continue
            
            # より近い衝突点があれば距離を更新
            distances = torch.minimum(distances, hit_distances)

        # 可視化が有効な場合、デバッグ用の線を描画
        if self.visualize:
            self._draw_debug_rays(ray_origins, world_ray_directions, distances)
            
        return distances

    def _ray_intersect_sphere(self, ray_origins, ray_directions, sphere_entity):
        sphere_pos = sphere_entity.get_pos().unsqueeze(1)
        radius = sphere_entity.morph.radius
        oc = ray_origins - sphere_pos
        a = torch.sum(ray_directions * ray_directions, dim=-1)
        b = 2.0 * torch.sum(oc * ray_directions, dim=-1)
        c = torch.sum(oc * oc, dim=-1) - radius * radius
        discriminant = b * b - 4 * a * c
        no_hit_mask = discriminant < 0
        t = (-b - torch.sqrt(torch.clamp(discriminant, min=0))) / (2.0 * a)
        t[t < 1e-4] = self.ray_length
        t[no_hit_mask] = self.ray_length
        return torch.clamp(t, max=self.ray_length)

    def _ray_intersect_box(self, ray_origins, ray_directions, box_entity):
        box_pos = box_entity.get_pos().unsqueeze(1)
        half_extents = torch.tensor(box_entity.morph.size, device=self.device) / 2.0
        min_bound = box_pos - half_extents
        max_bound = box_pos + half_extents
        t_min = (min_bound - ray_origins) / (ray_directions + 1e-6)
        t_max = (max_bound - ray_origins) / (ray_directions + 1e-6)
        t1 = torch.minimum(t_min, t_max)
        t2 = torch.maximum(t_min, t_max)
        t_near = torch.max(torch.max(t1[..., 0], t1[..., 1]), t1[..., 2])
        t_far = torch.min(torch.min(t2[..., 0], t2[..., 1]), t2[..., 2])
        hit_mask = (t_near < t_far) & (t_far > 1e-4)
        distances = torch.full_like(t_near, self.ray_length)
        distances[hit_mask] = t_near[hit_mask]
        return torch.clamp(distances, max=self.ray_length)
    
    def _ray_intersect_cylinder(self, ray_origins, ray_directions, cylinder_entity):
        cyl_pos = cylinder_entity.get_pos().unsqueeze(1)
        radius = cylinder_entity.morph.radius
        height = cylinder_entity.morph.height
        half_extents = torch.tensor([radius, radius, height / 2.0], device=self.device)
        min_bound = cyl_pos - half_extents
        max_bound = cyl_pos + half_extents
        t_min = (min_bound - ray_origins) / (ray_directions + 1e-6)
        t_max = (max_bound - ray_origins) / (ray_directions + 1e-6)
        t1 = torch.minimum(t_min, t_max)
        t2 = torch.maximum(t_min, t_max)
        t_near = torch.max(torch.max(t1[..., 0], t1[..., 1]), t1[..., 2])
        t_far = torch.min(torch.min(t2[..., 0], t2[..., 1]), t2[..., 2])
        hit_mask = (t_near < t_far) & (t_far > 1e-4)
        distances = torch.full_like(t_near, self.ray_length)
        distances[hit_mask] = t_near[hit_mask]
        return torch.clamp(distances, max=self.ray_length)

    def _draw_debug_rays(self, ray_origins, ray_directions, distances):
        self.scene.clear_debug_objects()
        """scene.draw_debug_line を使ってLIDARのレイを描画する。"""
        env_idx = self.vis_env_idx
        
        # 可視化対象の環境のデータを取得
        # ray_origins[env_idx] は shape (1, 3) なので、[0]でインデックスして shape (3,) のベクトルを取得
        start_point_numpy = ray_origins[env_idx, 0].cpu().numpy()
        directions_env = ray_directions[env_idx]
        distances_env = distances[env_idx]

        # ヒットした点またはレイの最大長の点を計算
        hit_distances = distances_env.unsqueeze(-1)
        # ray_origins[env_idx] の shape (1, 3) はブロードキャストされる
        end_points_env = ray_origins[env_idx] + directions_env * hit_distances
        
        # ヒットしたかどうかのマスク
        hit_mask = (distances_env < self.ray_length)
        
        # 色を定義
        green_color = [0.0, 1.0, 0.0]  # ミス
        red_color = [1.0, 0.0, 0.0]    # ヒット

        # 各レイについて線を描画
        for i in range(self.num_rays):
            # 開始点は全てのレイで共通
            start = start_point_numpy
            end = end_points_env[i].cpu().numpy()
            color = red_color if hit_mask[i] else green_color
            
            self.scene.draw_debug_line(
                start=start,
                end=end,
                color=color,
            )